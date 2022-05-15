import torch
import transformers


class CodeFormer(torch.nn.Module):
    """
    Implementation of CodeFormer model that shall serve for generating source code from
    a description in a natural language.
    """

    # Parameters that shall not be freezed when freezing BART model
    NON_FREEZABLE_PARAMETERS = [
        "model.encoder.embed_positions",
        "model.encoder.layers.0.self_attn.k_proj.weight",
        "model.encoder.layers.0.self_attn.k_proj.bias",
        "model.encoder.layers.0.self_attn.v_proj.weight",
        "model.encoder.layers.0.self_attn.v_proj.bias",
        "model.encoder.layers.0.self_attn.q_proj.weight",
        "model.encoder.layers.0.self_attn.q_proj.bias",
    ]

    def __init__(self, bart_model_path, encoder_model_path,
                 random_encoder=False,
                 encoder_decoder_dropout_prob=0.0,
                 decoder_dropout_prob=0.0,
                 random_init_last_x_layers_of_encoder=0,
                 use_relu=True):
        """
        Constructor
        :param bart_model_path: path to a pretrained BART model
        :param encoder_model_path: path to a pretrained encoder model or 'random'
        """
        super().__init__()

        self.__use_relu = use_relu

        # Load BART model pretrained for generating source code
        self.__bart_config = transformers.BartConfig.from_pretrained(bart_model_path)
        self.__bart_config.forced_bos_token_id = None  # Do not generate [START] automatically
        self.__bart_config.use_cache = False
        self.__bart_config.dropout = decoder_dropout_prob
        self.__bart_config.attention_dropout = decoder_dropout_prob
        self.__bart_config.activation_dropout = decoder_dropout_prob
        self.__bart_model: transformers.BartForConditionalGeneration = \
            transformers.BartForConditionalGeneration.from_pretrained(bart_model_path, config=self.__bart_config)

        # Load FERDA/CodeBERT model pretrained on the Stackoverflow
        if not random_encoder:
            self.__additional_encoder_config = transformers.AutoConfig.from_pretrained(encoder_model_path)
            self.__addition_encoder_model: transformers.AutoModel = \
                transformers.AutoModel.from_pretrained(encoder_model_path, config=self.__additional_encoder_config)

            if random_init_last_x_layers_of_encoder > 0:
                self.__init_x_last_layers_of_encoder(random_init_last_x_layers_of_encoder)
        else:
            additional_encoder_tok = transformers.AutoTokenizer.from_pretrained(encoder_model_path)
            self.__additional_encoder_config = transformers.BertConfig(vocab_size=additional_encoder_tok.vocab_size,
                                                                       hidden_size=768,
                                                                       num_hidden_layers=6,
                                                                       num_attention_heads=12)
            self.__addition_encoder_model = transformers.BertModel(config=self.__additional_encoder_config)

        # Linear layer that transforms the output of the FERDA model to the
        # input of the BART encoder
        self.__encoder_decoder_dropout = torch.nn.Dropout(p=encoder_decoder_dropout_prob)
        self.__dense = torch.nn.Linear(self.__additional_encoder_config.hidden_size, self.__bart_config.d_model)
        self.__relu = torch.nn.ReLU()

        # get used device
        self.__device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # check whether the model uses global attention mask or not
        self.__global_attention = False
        if isinstance(self.__addition_encoder_model, transformers.LongformerModel):
            self.__global_attention = True

    def __init_x_last_layers_of_encoder(self, x):
        """
        Initialize last x layers of the additional encoder by random values.
        This can be used as a regularization technique to prevent model's overfitting
        and additional encoder degradation.
        :param x: number of layers to initialize randomly
        :return: N/A
        """
        # Get layer number that shall be randomly initialized
        to_init = []
        for i in range(x):
            to_init.append(str(self.__addition_encoder_model.config.num_hidden_layers - i - 1))

        # Randomly initialize selected layers
        #   - ignore layers not in to_init
        #   - ignore layer_norm
        #   - use xavier normal for weights
        #   - use zeros or biases
        for name, layer in self.__addition_encoder_model.named_parameters():
            if "LayerNorm" in name:
                continue
            num = name[14:16]
            if num[-1] == ".":
                num = num[0]
            if num not in to_init:
                continue
            print(f"Doing random initialization of: {name}")
            if "weight" in name:
                torch.nn.init.xavier_normal_(layer)
            elif "bias" in name:
                torch.nn.init.zeros_(layer)

    def freeze_bart_base_model(self, freeze=True):
        """
        Freeze all parameters of the base BART model except the following:
            - positions embeddings
            - self-attention input projection matrix of the encoder's first layer

        NOTE: All weights of the additional encoder will remain trainable
        :param freeze: boolean flag indicating whether to freeze the BART's weights or not
        :return: N/A
        """
        requires_grad = not freeze
        for name, param in self.__bart_model.named_parameters():
            if name not in self.NON_FREEZABLE_PARAMETERS:
                param.requires_grad = requires_grad
            else:
                param.requires_grad = True

    def to_dict(self):
        """
        Creates a dictionary of the model's configuration
        :return: dictionary containing the configuration
        """
        config_dict = dict()
        for k, v in self.__additional_encoder_config.to_dict().items():
            config_dict[f"ferda_{k}"] = v

        for k, v in self.__bart_config.to_dict().items():
            config_dict[f"bart_{k}"] = v

        return config_dict

    def generate(self,
                 encoder_input,
                 encoder_attention_mask,
                 encoder_token_type_ids,
                 bos_token_id, pad_token_id, eos_token_id,
                 max_length=256, min_length=10,
                 num_beams=5,
                 num_predictions=1,
                 decoder_input_ids=None,
                 decoder_attention_mask=None
                 ):
        """
        Generate a text based on a given prompt using beam search algorithm
        :param encoder_input: IDs of the tokenized input sequence
        :param encoder_attention_mask: attention mask for the encoder (to mask out [SEP] and [PAD])
        :param encoder_token_type_ids: token type IDs for the encoder (0 for the textual part)
        :param bos_token_id: ID of the [START] token
        :param pad_token_id: ID of the [PAD] token
        :param eos_token_id: ID of the [END] token
        :param max_length: maximum length of the generated text
        :param min_length: minimum length of the generated text
        :param num_beams: number of beams to be used to generate the text
        :param num_predictions: number of candidate predictions to produce
        :param decoder_input_ids: decoder inputs
        :param decoder_attention_mask: decoder attention mask
        :return: TODO TBD.
        """

        # Build inputs for the additional encoder model
        inputs = {"input_ids": encoder_input,
                  "attention_mask": encoder_attention_mask,
                  "token_type_ids": encoder_token_type_ids,
                  }
        if self.__global_attention:  # global attention mask is only used for LongFormer
            # Create a global attention mask for the LongFormer model
            #   - enable the global attention to be calculated for [CLS] token only
            encoder_global_attention_mask = torch.zeros_like(encoder_attention_mask)
            encoder_global_attention_mask[:, 0] = 1
            encoder_global_attention_mask = encoder_global_attention_mask.to(self.__device)
            inputs["global_attention_mask"] = encoder_global_attention_mask

        # Calculate activations of the FERDA model
        input_embeddings = self.__addition_encoder_model(**inputs)[0]
        input_embeddings = self.__dense(input_embeddings)
        if self.__use_relu:
            input_embeddings = self.__relu(input_embeddings)
        input_embeddings = self.__encoder_decoder_dropout(input_embeddings)

        if decoder_input_ids is None:
            decoder_input_ids = torch.full((input_embeddings.shape[0], 1), bos_token_id)
            decoder_input_ids = decoder_input_ids.to(self.__device)

        return self.__bart_model.generate(max_length=max_length,
                                          min_length=min_length,
                                          early_stopping=True,
                                          num_beams=num_beams,
                                          bos_token_id=bos_token_id,
                                          eos_token_id=eos_token_id,
                                          pad_token_id=pad_token_id,
                                          inputs_embeds=input_embeddings,
                                          attention_mask=encoder_attention_mask,
                                          decoder_input_ids=decoder_input_ids,
                                          num_return_sequences=num_predictions,
                                          decoder_attention_mask=decoder_attention_mask
                                          )

    def forward(self,
                encoder_input,
                encoder_attention_mask,
                encoder_token_type_ids,
                labels,
                decoder_input,
                decoder_attention_mask,
                ):
        """
        Calculate a forward pass of the CodeFormer model
            -   takes in the ids of the input tokens (description in NL)
            -   outputs crossentropy loss between labels and the predicted code
        :param encoder_input: IDs of the tokenized input sequence
        :param encoder_attention_mask: attention mask for the encoder (to mask out [SEP] and [PAD])
        :param encoder_token_type_ids: token type IDs for the encoder (0 for the textual part)
        :param labels: expected IDs of the tokenized resulting source code
        :param decoder_input: input of the decoder (shall be same as labels shifted right)
        :param decoder_attention_mask: attention mask for the decoder (mask out [PAD] tokens,
                                       forward mask is added automatically)
        :return: Seq2SeqLMOutput
        """
        # Build inputs for the additional encoder model
        inputs = {"input_ids": encoder_input,
                  "attention_mask": encoder_attention_mask,
                  "token_type_ids": encoder_token_type_ids,
                  }
        if self.__global_attention:  # global attention mask is only used for LongFormer
            # Create a global attention mask for the LongFormer model
            #   - enable the global attention to be calculated for [CLS] token only
            encoder_global_attention_mask = torch.zeros_like(encoder_attention_mask)
            encoder_global_attention_mask[:, 0] = 1
            inputs["global_attention_mask"] = encoder_global_attention_mask

        # Calculate activations of the FERDA model
        input_embeddings = self.__addition_encoder_model(**inputs)[0]
        input_embeddings = self.__dense(input_embeddings)
        if self.__use_relu:
            input_embeddings = self.__relu(input_embeddings)
        input_embeddings = self.__encoder_decoder_dropout(input_embeddings)

        # Use FERDA's output as an input for the BART model
        bart_out = self.__bart_model(inputs_embeds=input_embeddings,  # FERDA's output embeds
                                     labels=labels,  # Correct output (dec. inputs shifted left)
                                     decoder_input_ids=decoder_input,  # Decoder's input
                                     attention_mask=encoder_attention_mask,  # Encoder input mask (same as for FERDA)
                                     decoder_attention_mask=decoder_attention_mask  # Decoder mask
                                     )

        return bart_out
