import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F


_tokenizer = _Tokenizer()

__all__ = ['multipf', 'MultiPF']


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")


    model = clip.build_model(state_dict)

    
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, device_id):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        self.device_id = device_id

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)  # n_cls, n_seq_len, dim
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    def obtain_label_emds(self, classnames):
        tokenized_names = torch.cat([clip.tokenize(name) for name in classnames])
        # class embedding token
        if torch.cuda.is_available():
                device = torch.device("cuda", self.device_id)
        else:
            device = torch.device("cpu")
        tokenized_names = tokenized_names.to(device)

        with torch.no_grad():
            
            embedding = self.token_embedding(tokenized_names).type(self.dtype)
            x = embedding + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)
            x = x.mean(dim=1)
            x = x @ self.text_projection
        # print("classnme embs:")
        # print(x.shape)

        return x  # n,d


class MLP(nn.Module):
    def __init__(self, input_dim, mid_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, mid_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mid_dim, output_dim)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, embeddings):
        
        output = self.linear1(embeddings)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        
        return output


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim, n_pool, epsilon=None, **kwargs):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim  # 512
        self.cond_dim = cond_dim  # 512
        self.n_pool = n_pool


        self.beta = nn.Parameter(torch.zeros(n_pool))
        self.gamma = nn.Parameter(torch.ones(input_dim))


        self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=n_pool, bias=False)
        self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        torch.nn.init.constant_(self.beta_dense.weight, 0)
        torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        for _ in range(len(inputs.shape) - len(cond.shape)):
            cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1) # cond （B,1,L,H）
        # print("cond")
        # print(cond.size())  # 6,1024
        beta = self.beta_dense(cond) + self.beta  # n_c,n_p  6,1024
        gamma = self.gamma_dense(cond) + self.gamma  # n_c,n_d

        outputs = inputs
        mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
        # n_p,1

        outputs = outputs - mean
        # n_p,n_d

        variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
        std = (variance + self.epsilon) ** 0.5
        outputs = outputs / std  # n_p,n_d 8,512

        outputs = torch.matmul(gamma, outputs.T)  # n_c, n_p
        print("outputs")
        print(outputs.size())
        outputs = outputs + beta

        return outputs  # n_c,n_p


class MultiPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        
        # prompt length
        n_ctx = cfg.TRAINER.PF_MLC.N_CTX
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        print("ccccccccccccccccc")
        print(ctx_dim)  # 512
        pool_size = cfg.TRAINER.PF_MLC.POOL_SIZE
        self.cln = cfg.CLN


        if self.cln:
            cln_dim = clip_model.text_projection.shape[1]
            print('clclclclclclcl')
            print(cln_dim)
            self.layer_norm = LayerNorm(input_dim=ctx_dim, cond_dim=cln_dim, n_pool=pool_size)
        else:
            cln_dim = clip_model.text_projection.shape[1]
            self.pc_layer = nn.Linear(cln_dim, ctx_dim)

     
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"


        # prompt_pool_shape = (pool_size, n_ctx, ctx_dim)
        print("Initializing a generic context")
        ctx_vectors = torch.empty(pool_size, n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Size of prompt pool: {pool_size}")
        print(f"Number of context words (tokens): {n_ctx}")

        self.prompt = nn.Parameter(ctx_vectors)  # to be optimized

        self.n_ctx = n_ctx
        self.class_token_position = 'end'
        self.token_embedding = clip_model.token_embedding
        self.device_id = cfg.TRAINER.DEVICEID

    def forward(self, classnames, name_embs):
        
        n_cls = len(classnames)
        prompt_prefix = " ".join(["X"] * self.n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        
        if torch.cuda.is_available():
            device = torch.device("cuda", self.device_id)
        else:
            device = torch.device("cpu")
        tokenized_prompts = tokenized_prompts.to(device)

        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)
        
        
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.n_ctx:, :]
        
        prompt = self.prompt.to(device)  # (pool_size, n_ctx, ctx_dim)

        if self.cln:
            prompt_avg = torch.mean(prompt, dim=-2)
            weights = self.layer_norm(prompt_avg, name_embs.to(device))  # n_cls,n_p
            # print(weights)
        else:
            prompt_avg = torch.mean(prompt, dim=-2)
            prompt_avg = prompt_avg / prompt_avg.norm(dim=-1, keepdim=True)
            # print(prompt_avg)
            cls_p = self.pc_layer(name_embs.to(device))  # n_cls,n_p
            # print(cls_p)
            cls_p = cls_p / cls_p.norm(dim=-1, keepdim=True)

            weights = cls_p @ prompt_avg.T  # n_cls, pool_size


        weights = F.softmax(weights, dim=-1)  # n_cls, pool_size
        prompt = prompt.permute(2, 0, 1)  # dim, pool_size, length
        weights = weights.unsqueeze(dim=0)  # 1, n_cls, pool_size

        weighted_prompts = torch.matmul(weights, prompt)  # dim, n_cls, length
        # (1, n_cls, pool_size) × (dim, pool_size, length) = （dim, n_cls, length）

        weighted_prompts = weighted_prompts.permute(1, 2, 0)   # n_cls, length, dim

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    weighted_prompts,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        else:
            raise ValueError

        return prompts, tokenized_prompts, weights


class PFA_Block_3d(nn.Module):

    def __init__(self, input_dim, latent_dim, layers_num, dropout_ratio=0.5):

        super(PFA_Block_3d, self).__init__()

        self.layers_num = layers_num
        self.latent_dim = latent_dim
        self.dropout_ratio = dropout_ratio
        # print('1111------------------------------pfadropout---------------')
        # print(dropout_ratio)

        #
        # Build the network.
        #
        layers = [
            nn.Linear(2 * input_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(self.layers_num):
            layers.extend(
                [
                    nn.BatchNorm1d(self.latent_dim),
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            # print('222------------------------------pfadropout---------------')
            # print(dropout_ratio)
            if dropout_ratio > 0:
                layers.append(nn.Dropout(p=dropout_ratio))
                # print('333------------------------------pfadropout---------------')
                # print(dropout_ratio)


        layers.extend(
            [
                nn.Linear(latent_dim, input_dim),
                #nn.LeakyReLU(0.2, inplace=True)
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # print(self.dropout_ratio)
        # print('111111111111111111')
        ab = torch.cat((a, b), dim=-1)

        ab_size = ab.size()
        ab = ab.view(-1, ab_size[-1])  # Reshape to (batch_size * 30, 1024)
        print(ab.size())

        out = self.net(ab)
        out = out / out.norm(dim=-1, keepdim=True)

        # Reshape back to (batch_size, 30, 1024)
        out = out.view(ab_size[0], ab_size[1], -1)

        # out = self.net(ab)

        return out


class PFAModule(nn.Module):
    def __init__(
            self,
            input_dim, dropout,
            latent_dim, layers_num,
            **kwds):

        super(PFAModule, self).__init__()
        self.union_op = PFA_Block_3d(input_dim=input_dim, latent_dim=latent_dim, layers_num=layers_num, dropout_ratio=dropout)

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        a_U_b = self.union_op(a, b)
        b_U_a = self.union_op(b, a)

        return a_U_b, b_U_a





class MultiPF(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()

        
        self.prompt_learner = MultiPromptLearner(cfg, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, cfg.TRAINER.DEVICEID)
        self.image_encoder_pfa = clip_model.visual_pfa
        self.device = cfg.TRAINER.DEVICEID
        
        self.dtype = clip_model.dtype


        self.count_mlp = MLP(input_dim=1024, mid_dim=300, output_dim=4)

        self.visual_encoder_type = cfg.MODEL.BACKBONE.NAME

        # ALPHA
        self.learnable_alpha = cfg.LEARN_ALPHA
        if self.learnable_alpha:
            self.alpha = nn.Parameter(torch.ones(2))

        # uncertainty weight
        self.uncertainty_weight = cfg.UNCERTAIN_WEIGHT
        if self.uncertainty_weight:
            if cfg.USE_PFA:
                self.loss_weight = nn.Parameter(torch.ones(5))
            else:
                self.loss_weight = nn.Parameter(torch.ones(3))

        # 3dpfa
        self.use_pfa = cfg.USE_PFA
        self.pfa_layer_num = cfg.PFA_LAYER_NUM
        self.pfa_layer_dim = cfg.PFA_LAYER_DIM
        self.pfa_dropout = cfg.PFA_DROPOUT

        if self.use_pfa:
            self.pfa_model = PFAModule(input_dim=1024, dropout=self.pfa_dropout,
                                           latent_dim=self.pfa_layer_dim, layers_num=self.pfa_layer_num)


    def forward(self, classnames, image):
        # get image and text features
        image_features = self.image_encoder(image.type(self.dtype))

        # print(image_features.size())  # 30,1024
        image_features_3d, image_features_k = self.image_encoder_pfa(image.type(self.dtype))

        image_features_3d = image_features_3d.permute(0, 2, 1).to(torch.float32)
        image_features_k = image_features_k.permute(1, 0, 2).to(torch.float32)  # 30, 50, 2048
        # print(image_features_3d.size())  # 30, 50, 1024

        name_embs = self.text_encoder.obtain_label_emds(classnames)

        prompts, tokenized_prompts, weights = self.prompt_learner(classnames, name_embs)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        # print("image features: ", image_features.shape)  # n_cls*n_num,dim
        # print("text features: ", text_features.shape)  # n_cls,dim

        count_model = self.count_mlp

        class_embs = name_embs

        # normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        loss_weight = None
        alpha = 0
        pfa_model = None

        if self.learnable_alpha:
            alpha = torch.exp(self.alpha[0]) / torch.sum(torch.exp(self.alpha))


        if self.uncertainty_weight:
            loss_weight = self.loss_weight
        if self.use_pfa:
            pfa_model = self.pfa_model

        return image_features, text_features, count_model, weights, alpha, loss_weight, pfa_model, class_embs, image_features_3d, image_features_k  # , self.predict_c

    @property
    def network_name(self):
        name = ''
        name += 'MultiPF-{}'.format(self.visual_encoder_type)
        return name

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            print(name)
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
                print(name)
        return params

    def prompt_params(self):
        params = []
        print("Prompt params: ")
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                print(name)
                params.append(param)

        return params
    
    def mlpcount_params(self):
        params = []
        print("Count params: ")
        for name, param in self.named_parameters():
            if 'count_mlp' in name:
                # print(name)
                params.append(param)
        return params

    def learnable_params(self):
        params = []
        print("alpha and loss_learnable params: ")
        for name, param in self.named_parameters():
            if "loss_weight" in name:
                params.append(param)
            elif "alpha" in name:
                params.append(param)
        return params

    def pfa_params(self):
        params = []
        print("alpha and loss_learnable params: ")
        for name, param in self.named_parameters():
            if "pfa_model" in name:
                params.append(param)
        return params


def multipf(cfg, **kwargs):
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building multipf")
    model = MultiPF(cfg, clip_model)

    if not cfg.TRAINER.FINETUNE_BACKBONE:
        print('Freeze the backbone weights')
        backbone_params = model.backbone_params()
        for param in backbone_params:
            param.requires_grad_(False)

    if not cfg.TRAINER.FINETUNE_ATTN:
        print('Freeze the attn weights')
        attn_params = model.attn_params()
        for param in attn_params:
            param.requires_grad_(False)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda", cfg.TRAINER.DEVICEID)
    else:
        device = torch.device("cpu")
    model.to(device)

    
    return model
