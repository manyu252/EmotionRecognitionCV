from emotion_recognition import select_model
from torchsummary import summary
import sys

model = select_model(sys.argv[1])
summary(model, (3, 48, 48), device="cpu", batch_size=1024)

