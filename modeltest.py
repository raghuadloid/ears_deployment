from tensorflow.keras.models import load_model

model = load_model("./bestmodel_2ears.h5", compile=False)

print(model.summary())
