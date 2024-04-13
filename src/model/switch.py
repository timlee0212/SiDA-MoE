import copy
import os
import warnings
from typing import Optional, Tuple, Union

import accelerate
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    SwitchTransformersConfig,
    SwitchTransformersForConditionalGeneration,
)
from transformers.models.switch_transformers.modeling_switch_transformers import (
    __HEAD_MASK_WARNING_MSG,
    MoEModelOutput,
    Seq2SeqMoEOutput,
    SwitchTransformersEncoderModel,
    SwitchTransformersPreTrainedModel,
    load_balancing_loss_func,
    router_z_loss_func,
)

import model.soda_moe as soda_moe
from model.soda_moe import SwitchTransformersStack


class SwitchTransformersForConditionalGenerationOffload(SwitchTransformersForConditionalGeneration):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        encoder_config.is_offloading = True
        encoder_config.eval_batch_size = 64
        self.encoder = SwitchTransformersStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        decoder_config.is_offloading = False
        self.decoder = SwitchTransformersStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        hash_table: Optional[dict] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqMoEOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
        >>> model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-8")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> # . To, let’s say you have a dog. To summarize:
        >>> # Since the model has been trained on MLM, this will output gibberish
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                return_dict=return_dict,
                hash_table=hash_table,
            )
        elif return_dict and not isinstance(encoder_outputs, MoEModelOutput):
            encoder_outputs = MoEModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                router_probs=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        encoder_z_loss = None
        encoder_aux_loss = None
        decoder_z_loss = None
        decoder_aux_loss = None

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            # todo check in the config if router loss enables

            if output_router_logits:
                # Compute the router loss (z_loss + auxiliary loss) for each router in the encoder and decoder
                encoder_router_logits, encoder_expert_indexes = self._unpack_router_logits(encoder_outputs.router_probs)
                encoder_z_loss = router_z_loss_func(encoder_router_logits)
                encoder_router_probs = nn.Softmax(dim=-1)(encoder_router_logits)
                encoder_aux_loss = load_balancing_loss_func(encoder_router_probs, encoder_expert_indexes)

                decoder_router_logits, decoder_expert_indexes = self._unpack_router_logits(decoder_outputs.router_probs)
                decoder_z_loss = router_z_loss_func(decoder_router_logits)
                decoder_router_probs = nn.Softmax(dim=-1)(decoder_router_logits)
                decoder_aux_loss = load_balancing_loss_func(decoder_router_probs, decoder_expert_indexes)

            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            if output_router_logits and labels is not None:
                z_loss = self.router_z_loss_coef * (encoder_z_loss + decoder_z_loss)
                aux_loss = self.router_aux_loss_coef * (encoder_aux_loss + decoder_aux_loss)
                loss = loss + z_loss + aux_loss

        if not return_dict:
            output = (lm_logits,)
            if output_router_logits:  # only return the loss if they are not None
                output += (
                    encoder_z_loss,
                    encoder_aux_loss,
                    decoder_z_loss,
                    decoder_aux_loss,
                    *decoder_outputs[1:],
                    *encoder_outputs,
                )
            else:
                output += (*decoder_outputs[1:], *encoder_outputs)

            return ((loss,) + output) if loss is not None else output
        return Seq2SeqMoEOutput(
            loss=loss,
            logits=lm_logits,
            encoder_z_loss=encoder_z_loss,
            encoder_aux_loss=encoder_aux_loss,
            decoder_z_loss=decoder_z_loss,
            decoder_aux_loss=decoder_aux_loss,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_router_logits=encoder_outputs.router_probs,
            decoder_router_logits=decoder_outputs.router_probs,
        )


class SwitchTransformersClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: SwitchTransformersConfig, num_labels, classifier_dropout=0.0):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# Define a custom model that uses BERT and a linear classification layer
class SwitchTransformersClassificationModel(SwitchTransformersPreTrainedModel):
    def __init__(
        self,
        num_labels,
        MODEL,
        BASEDIR,
        tutel=False,
        meta=False,
        pretrained=True,
        offloading=False,
        classifier_dropout=0.0,
        degrad_factor=None,
    ):
        config = AutoConfig.from_pretrained(f"google/{MODEL}", cache_dir=f"{BASEDIR}/tmp/")
        super(SwitchTransformersClassificationModel, self).__init__(config)
        config.is_offloading = offloading
        self.is_offloading = offloading
        config.tutel = tutel
        if degrad_factor is not None:
            config.degrading_factor = degrad_factor

        self.switch_transformers = (
            soda_moe.SwitchTransformersEncoderModel if offloading else SwitchTransformersEncoderModel
        )
        if meta:
            with accelerate.init_empty_weights():
                self.switch_transformers = self.switch_transformers.from_pretrained(
                    f"google/{MODEL}", cache_dir=f"{BASEDIR}/tmp/", config=config
                )
        elif pretrained:
            self.switch_transformers = self.switch_transformers.from_pretrained(
                f"google/{MODEL}", cache_dir=f"{BASEDIR}/tmp/", config=config
            )
        else:
            self.switch_transformers = self.switch_transformers(config)

        self.classifier = SwitchTransformersClassificationHead(
            config, num_labels, classifier_dropout=classifier_dropout
        )
        self.num_labels = num_labels
        self.loss = nn.CrossEntropyLoss()

    def gradient_checkpointing_enable(self):
        self.switch_transformers.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.switch_transformers.gradient_checkpointing_disable()

    @classmethod
    def from_checkpoint(cls, num_labels, MODEL, BASEDIR, ckpt_path, offloading=False, **kwargs):
        assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} not found."
        model = cls(num_labels, MODEL, BASEDIR, pretrained=False, offloading=offloading, **kwargs)
        model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
        return model

    def forward(self, input_ids, attention_mask, labels=None, hash_table=None):
        if self.is_offloading:
            outputs = self.switch_transformers(input_ids=input_ids, hash_table=hash_table)
        else:
            outputs = self.switch_transformers(input_ids=input_ids)

        pooler_output = outputs.last_hidden_state.mean(1)
        logits = self.classifier(pooler_output)
        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits


class SwitchTransformersClassificationModel_Multirc(nn.Module):
    def __init__(
        self,
        num_labels,
        MODEL,
        BASEDIR,
        tutel=False,
        meta=False,
        pretrained=True,
        offloading=False,
        classifier_dropout=0.0,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(f"google/{MODEL}", cache_dir=f"{BASEDIR}/tmp/")
        config.is_offloading = offloading
        self.is_offloading = offloading
        config.tutel = tutel

        self.switch_transformers = (
            soda_moe.SwitchTransformersEncoderModel if offloading else SwitchTransformersEncoderModel
        )
        if meta:
            with accelerate.init_empty_weights():
                self.switch_transformers = self.switch_transformers.from_pretrained(
                    f"google/{MODEL}", cache_dir=f"{BASEDIR}/tmp/", config=config
                )
        elif pretrained:
            self.switch_transformers = self.switch_transformers.from_pretrained(
                f"google/{MODEL}", cache_dir=f"{BASEDIR}/tmp/", config=config
            )
        else:
            self.switch_transformers = self.switch_transformers(config)

        self.classifier = SwitchTransformersClassificationHead(
            config, num_labels, classifier_dropout=classifier_dropout
        )
        self.num_labels = num_labels
        self.loss = nn.CrossEntropyLoss()

    @classmethod
    def from_checkpoint(cls, num_labels, MODEL, BASEDIR, ckpt_path, offloading=False):
        assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} not found."
        model = cls(num_labels, MODEL, BASEDIR, pretrained=False, offloading=offloading)
        model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
        return model

    def forward(self, input_ids, attention_mask, labels=None, hash_table=None, idx=None):
        if self.is_offloading:
            outputs = self.switch_transformers(input_ids=input_ids, hash_table=hash_table)
        else:
            outputs = self.switch_transformers(input_ids=input_ids)

        pooler_output = outputs.last_hidden_state.mean(1)
        logits = self.classifier(pooler_output)
        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
            return {'loss': loss, 'logits': logits, 'idx': idx}
        return logits