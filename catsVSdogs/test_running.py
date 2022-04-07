from keras.preprocessing.image import ImageDataGenerator
from model_maker import define_model
from diagnostics import summarize_diagnostics


def run_test_harness():
	model = define_model()
	datagen = ImageDataGenerator(featurewise_center=True)
	datagen.mean = [123.68, 116.779, 103.939]
	train_it = datagen.flow_from_directory('Cats vs. Dogs/training_set/',
										   class_mode='binary', batch_size=64, target_size=(224, 224))
	test_it = datagen.flow_from_directory('Cats vs. Dogs/test_set/',
										  class_mode='binary', batch_size=64, target_size=(224, 224))
	history = model.fit(train_it, steps_per_epoch=len(train_it),
								  validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)
	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	summarize_diagnostics(history)
	model.save('final_model.h5')


if __name__ == "__main__":
	run_test_harness()
