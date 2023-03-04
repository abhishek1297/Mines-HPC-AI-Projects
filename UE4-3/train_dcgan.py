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
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input) 

def step(netD, netG, data, criterion, optimizerD, optimizerG, device):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    netD.zero_grad()        
    # Format batch
    real_image = data[0].to(device)
    b_size = real_image.size(0)
    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
    # Forward pass real batch through D
    output = netD(real_image).view(-1)
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    errD_real.backward()
    real_score = output.mean().item()
    
    ## Train with all-fake batch
    # Generate batch of latent vectors
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients    
    noise = torch.randn(b_size, CFG.n_z, 1, 1, device=device)
    # Generate fake image batch with G
    fake = netG(noise)
    label.fill_(fake_label)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    errD_fake.backward()
    fake_score = output.mean().item()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()    
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = netD(fake).view(-1)
    # Calculate G's loss based on this output
    errG = criterion(output, label)
    # Calculate gradients for G
    errG.backward()
    gen_score = output.mean().item()
    # Update G
    optimizerG.step()

    
    return errD, errG, real_score, fake_score, gen_score


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
    if model_disc_path:
        netD.load_state_dict(torch.load(model_disc_path))
        
    if verbose:
        print(netG)
        print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

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

            errD, errG, real_score, fake_score, gen_score = step(netD, netG, data, criterion,
                                                                 optimizerD, optimizerG, device)
            # if verbose and i % 50 == 0:
            #     print('[%d/%d][%d/%d]\tloss_D: %.4f\tloss_G: %.4f\treal_score: %.4f\tfake_score: %.4f / %.4f'
            #           % (epoch, CFG.n_epochs, i, len_,
            #              errD.item(), errG.item(), real_score, fake_score, gen_score))
        writer.add_scalars("train-loss", {"discriminator": errD.item(),
                                          "generator": errG.item()},
                           epoch)
        writer.add_scalars("train-score", {"disc-real": real_score,
                                            "disc-fake": fake_score},
                           epoch)
        if epoch % 20 == 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                grid = denorm(make_grid(fake, padding=2))
                writer.add_image(f"gen-image", grid, epoch)
                fake_fname = os.path.join(GEN_IMG_DIR, f"gen-{epoch}.png")
                save_image(grid, fake_fname, nrow=8)
            model_fname1 = os.path.join(MODEL_DIR, f"dcgan-disc-{epoch}.pth")
            model_fname2 = os.path.join(MODEL_DIR, f"dcgan-gen-{epoch}.pth")
            torch.save(netD.state_dict(), model_fname1)
            torch.save(netG.state_dict(), model_fname2)
            
        
        # Validation set metrics
        netD.eval()
        netG.eval()
        for i, data in enumerate(val_dataloader, 0):
            errD, errG, real_score, fake_score, gen_score = step(netD, netG, data, criterion,
                                                                 optimizerD, optimizerG, device)
        writer.add_scalars("val-loss", {"discriminator": errD.item(),
                                          "generator": errG.item()},
                           epoch)
        writer.add_scalars("val-score", {"disc-real": real_score,
                                            "disc-fake": fake_score},
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