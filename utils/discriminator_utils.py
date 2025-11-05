import torch


def compute_r1_penalty(
    discriminator,
    real_samples,
    class_vector,
    cont_features,
    x_t,
    t2=None,
    t1=None,
):
    """Compute the R1 regularization term on real samples.

    The R1 penalty encourages the discriminator to have small gradients with respect to
    its real inputs, which improves stability compared to the traditional gradient
    penalty that mixes real and fake samples.
    """
    with torch.enable_grad():
        real_samples = real_samples.detach().requires_grad_(True)
        real_logits = discriminator(
            real_samples,
            x_t,
            class_vector,
            cont_features,
            t2,
            t1,
            fake_input=False,
        )

        grad_outputs = torch.ones_like(real_logits)
        gradients = torch.autograd.grad(
            outputs=real_logits,
            inputs=real_samples,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]

        if gradients is None:
            return torch.tensor(0.0, device=real_samples.device)

        gradients = gradients.view(gradients.size(0), -1)
        penalty = 0.5 * gradients.pow(2).sum(dim=1)
        return penalty.mean()
