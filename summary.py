from Emo5CNN import Emo5CNN
from torchsummary import summary

model = Emo5CNN()
summary(model, (3, 48, 48), device="cpu", batch_size=1024)

