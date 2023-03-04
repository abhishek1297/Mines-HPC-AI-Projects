from common import *
# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(CFG.n_z, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, CFG.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(CFG.n_channels, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input) 
    

def gradient_penalty(netD, real_image, fake_image, batch_size, device):
    #alpha is selected randomly between 0 and 1
    alpha = torch.rand(batch_size,1,1,1).repeat(1, CFG.n_channels, CFG.image_size, CFG.image_size)
    alpha = alpha.to(device)
    # print(alpha.shape, real_image.shape, fake_image.shape)
    # interpolated image=randomly weighted average between a real and fake image
    # interpolated image ← alpha *real image  + (1 − alpha) fake image
    interpolated_image = (alpha * real_image) + (1 - alpha) * fake_image
    
    # calculate the critic score on the interpolated image
    interpolated_score = netD(interpolated_image)
    
    # take the gradient of the score wrt to the interpolated image
    gradient = torch.autograd.grad(inputs=interpolated_image,
                                  outputs=interpolated_score,
                                  retain_graph=True,
                                  create_graph=True,
                                  grad_outputs=torch.ones_like(interpolated_score) 
                                 )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    
    return gradient_penalty
    
def step(netD, netG, data, optimizerD, optimizerG, device):
    ############################
    # (1) Update D network
    ###########################
    ## Train with all-real batch
    netD.zero_grad()        
    # Format batch
    real_image = data[0].to(device)
    b_size = real_image.size(0)
    # train discriminator for critic_iters times
    for _ in range(CFG.critical_iters):
        noise = torch.randn(b_size, CFG.n_z, 1, 1, device=device)
        # Forward pass real batch through D
        output = netD(real_image).view(-1)
        real_mean = output.mean()
        fake = netG(noise)
        output = netD(fake.detach()).view(-1)
        fake_mean = output.mean()
        # Compute error of D
        errD = -real_mean + fake_mean
        if CFG.APPLY_GRAD_PENALTY:
            gp = gradient_penalty(netD, real_image, fake, b_size, device)
            errD += (CFG.lambda_gp * gp)
        errD.backward(retain_graph=True)
        # Update D
        optimizerD.step()
    
    if not CFG.APPLY_GRAD_PENALTY:
        # clip critic weights between -0.01, 0.01
        for p in netD.parameters():
            p.data.clamp_(-CFG.weight_clip, CFG.weight_clip)
    
    
    ############################
    # (2) Update G network
    ###########################
    netG.zero_grad()    
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake).view(-1)
    # Calculate G's loss based on this output
    errG = output.mean()
    if CFG.APPLY_GRAD_PENALTY:
        errG = -errG
    # Calculate gradients for G
    errG.backward()
    # Update G
    optimizerG.step()

    
    return errD, errG


def run(verbose, writer, device, model_gen_path=None, model_disc_path=None):
    # Create the generator and discriminator
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)
    netD.apply(weights_init)
    # Print the model
    if model_gen_path:
        netG.load_state_dict(torch.load(model_gen_path))
        print("loaded generator model")
    if model_disc_path:
        netD.load_state_dict(torch.load(model_disc_path))
        print("loaded discriminator model")
        
    if verbose:
        print(netG)
        print(netD)
    # exit()
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, CFG.n_z, 1, 1, device=device)


    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=CFG.lr_d, betas=(CFG.beta1, CFG.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=CFG.lr_g, betas=(CFG.beta1, CFG.beta2))
    
    train_dataloader, val_dataloader = load_dataset()
    len_ = len(train_dataloader)
    
    for epoch in tqdm(range(1, CFG.n_epochs+1)):
    
        # For each batch in the dataloader
        netD.train()
        netG.train()
        for i, data in enumerate(train_dataloader, 0):

            errD, errG = step(netD, netG, data, optimizerD, optimizerG, device)
        writer.add_scalars("train-loss", {"discriminator": errD.item(),
                                          "generator": errG.item()},
                           epoch)
        if epoch % CFG.save_every_n == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                grid = denorm(make_grid(fake, padding=2))
                writer.add_image(f"gen-image", grid, epoch)
                fake_fname = os.path.join(GEN_IMG_DIR, f"gen-{epoch}.png")
                save_image(grid, fake_fname, nrow=8)
            f1 = f"wgan-disc-{epoch}.pth"
            f2 = f"wgan-gen-{epoch}.pth"
            if CFG.APPLY_GRAD_PENALTY:
                f1 = "gp-" + f1
                f2 = "gp-" + f2
            else:
                f1 = "wc-" + f1
                f1 = "wc-" + f1
            model_fname1 = os.path.join(MODEL_DIR, f1)
            model_fname2 = os.path.join(MODEL_DIR, f2)
            torch.save(netD.state_dict(), model_fname1)
            torch.save(netG.state_dict(), model_fname2)
            
        
        # Validation set metrics
        netD.eval()
        netG.eval()
        for i, data in enumerate(val_dataloader, 0):
            errD, errG = step(netD, netG, data, optimizerD, optimizerG, device)
        writer.add_scalars("val-loss", {"discriminator": errD.item(),
                                          "generator": errG.item()},
                           epoch)
        
    writer.flush()
    writer.close()

if __name__ == "__main__":
    model_disc_path = CFG.model_disc_path
    model_gen_path = CFG.model_gen_path
    verbose = 0
    writer = SummaryWriter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("training")
    st = time()
    run(verbose, writer, device, model_gen_path, model_disc_path)
    t = time() - st
    td = timedelta(seconds=t)
    print("\n\nfinished in", td)
    torch.cuda.empty_cache()