from sys import exit

def train(
    GeneratorX,
    GeneratorY,
    DiscriminatorX,
    DiscriminatorY,
    opt_generator,
    opt_discriminator,
    epoch,
    loader,
    config,
    g_scaler=None,
    d_scaler=None,
):
    import tqdm
    import torch
    import random

    loop = tqdm.tqdm(loader, leave=True)
    mse_loss_fn = torch.nn.MSELoss()
    for idx, (voice1, sample_rate1, voice2, sample_rate2) in enumerate(loop):
        
        voice1 = voice1.to(config.device)
        voice2 = voice2.to(config.device)

        # train the discriminator
        opt_discriminator.zero_grad()

        fake_voice1 = GeneratorX(voice2)
        discr_real_voice1 = DiscriminatorX(voice1)
        discr_fake_voice1 = DiscriminatorX(fake_voice1.detach())
        loss_voice1 = (
            mse_loss_fn((discr_fake_voice1), torch.ones_like(discr_fake_voice1)) +
            mse_loss_fn((discr_real_voice1), torch.zeros_like(discr_real_voice1))
        )

        fake_voice2 = GeneratorY(voice1)
        discr_real_voice2 = DiscriminatorY(voice2)
        discr_fake_voice2 = DiscriminatorY(fake_voice2.detach())
        loss_voice2 = (
            mse_loss_fn((discr_fake_voice2), torch.ones_like(discr_fake_voice2)) + 
            mse_loss_fn((discr_real_voice2), torch.zeros_like(discr_real_voice2))
        )

        discr_loss = (loss_voice1 + loss_voice2) / 2
        discr_loss.backward()
        opt_discriminator.step()

        # train the generator
        opt_generator.zero_grad()

        # adversial loss
        discr_fake_voice1 = DiscriminatorX(fake_voice1)
        discr_fake_voice2 = DiscriminatorX(fake_voice2)
        loss_re_voice1 = mse_loss_fn(discr_fake_voice1, torch.ones_like(discr_fake_voice1))
        loss_re_voice2 = mse_loss_fn(discr_fake_voice2, torch.ones_like(discr_fake_voice2))
        gen_adv_loss = loss_re_voice1 + loss_re_voice2

        # cycle consistency loss
        inv_fake_voice1 = GeneratorY(fake_voice1)
        inv_fake_voice2 = GeneratorY(fake_voice2)
        loss_inv_voice1 = torch.linalg.norm(inv_fake_voice1, voice1, p=1, keepdim=True)
        loss_inv_voice2 = torch.linalg.norm(inv_fake_voice2, voice2, p=1, keepdim=True)
        cyc_loss = loss_inv_voice1 + loss_inv_voice2

        # identity loss
        ident_loss_voice1 = torch.linalg.norm(GeneratorX(voice1), voice1)
        ident_loss_voice2 = torch.linalg.norm(GeneratorY(voice2), voice2)
        ident_loss = ident_loss_voice1 + ident_loss_voice2

        gen_loss = gen_adv_loss + config.lambda_cyc * cyc_loss + config.lambda_id * ident_loss
        gen_loss.backward()
        opt_generator.step()

        if not idx%10:
            rnd_idx = random.randint(0, config.batch_size)
            save_audio(fake_voice1[rnd_idx], sample_rate1[rnd_idx], path=f"results/samples/{epoch}_fake_voice1.wav")
            save_audio(fake_voice2[rnd_idx], sample_rate2[rnd_idx], path=f"results/samples/{epoch}_fake_voice2.wav")

    if not epoch%1:
        torch.save({
            "gen_X2Y": GeneratorY.state_dict(),
            "gen_Y2X": GeneratorX.state_dict(),
            "opt_discr": opt_discriminator.state_dict(),
            "opt_gen": opt_generator.state_dict(),
            "epoch": epoch,
        }, f"resuls/checkpoints/cp_{epoch}.pth")


def save_audio(waveform, sample_rate, path):
    from torchaudio.transforms import InverseSpectrogram

    transform = InverseSpectrogram(n_mels=24, normalized=True)
    inv_waveform = transform(waveform, sample_rate)
    return inv_waveform


def pad_sequence(sequences, batch_first=False, padding_value=0):
    """Copied from: https://github.com/pytorch/pytorch/blob/8e93159fb65edba0934da76772ea4f77d3590156/torch/nn/utils/rnn.py#L325-L344
    
    I've modified the original function to work in the 2D case.
    """

    max_size = sequences[0].size()
    trailing_dims = max_size[1:]

    max_len = max([s.size(-1) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        time = tensor.size(-1)
        out_tensor[i, ..., :time] = tensor

    return out_tensor

def plot_specgram(waveform, sample_rate):
    import matplotlib.pyplot as plt

    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    plt.show()
