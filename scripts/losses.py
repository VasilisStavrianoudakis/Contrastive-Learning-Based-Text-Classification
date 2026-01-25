import torch
import torch.nn as nn
from torch.nn import functional as F


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    """Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations"""

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
    ):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = (
            self.asymmetric_w
        ) = self.loss = None

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    """
    This loss is intended for single-label classification problems
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction="mean"):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        """
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1, target.long().unsqueeze(1), 1
        )

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(
                self.eps / num_classes
            )

        # loss calculation
        loss = -self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()
        return loss


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning:
        https://arxiv.org/abs/2004.11362

        Args:
            temperature (float, optional). Defaults to 0.07.
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """

        Args:
            embeddings (torch.Tensor): Transformer model output.
            labels (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: Loss.
        """
        device = torch.device("cuda") if embeddings.is_cuda else torch.device("cpu")
        # We have to l2 normalize our embeddings when computing cosine similarity.
        embeddings = F.normalize(embeddings, p=2, dim=1)

        dot_product_tempered = torch.mm(embeddings, embeddings.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(
                dot_product_tempered
                - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]
            )
            + 1e-5
        )

        mask_similar_class = (
            labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels
        ).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]

        # cardinality_per_samples = torch.sum(mask_combined, dim=1)
        # print("cardinality_per_samples", cardinality_per_samples)

        cardinality_per_samples = mask_combined.sum(1)
        cardinality_per_samples = torch.where(
            cardinality_per_samples < 1e-6, 1, cardinality_per_samples
        )

        log_prob = -torch.log(
            exp_dot_tempered
            / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True))
        )
        supervised_contrastive_loss_per_sample = (
            torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        )
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss


class SCL(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning:
        https://arxiv.org/abs/2011.01403

        Args:
            temperature (float, optional). Defaults to 0.07.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """

        Args:
            embeddings (torch.Tensor): Transformer model output.
            labels (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: Loss.
        """
        # We have to l2 normalize our embeddings when computing cosine similarity.
        embeddings = F.normalize(embeddings, p=2, dim=1)

        cosine_sim = torch.mm(embeddings, embeddings.T) / self.temperature
        cosine_sim = torch.exp(cosine_sim)

        # Calculate row sum, without the main diagonal.
        row_sum = torch.sum(cosine_sim.clone().fill_diagonal_(0), dim=1)

        # Outer sum.
        losses = torch.zeros(size=(embeddings.shape[0],))
        for i in range(embeddings.shape[0]):
            n_i = labels[labels == labels[i]].shape[0] - 1
            if n_i == 0:
                continue
            inner_sum = 0
            # Calculate inner sum.
            for j in range(embeddings.shape[0]):
                if labels[i] == labels[j] and i != j:
                    inner_sum = inner_sum + torch.log(cosine_sim[i][j] / row_sum[i])

            losses[i] = inner_sum / (-n_i)

        # NOTE: If our label values are all different inside a batch, we won't be able to form positive pairs.
        # In that case, we won't be able to compute the loss, so return loss=0.
        if losses[losses != 0].shape[0] == 0:  # FIXME: We can check if empty better.
            contrastive_loss = torch.tensor(
                0.0, device=embeddings.device, requires_grad=True
            )
        else:
            contrastive_loss = torch.mean(losses[losses != 0])

        return contrastive_loss
