# sequence_classification_deep_learning
For study Sequence Classification Using Deep Learning

## Create inputs from randoms series ##

#### Series generator ####

```
def function_serie_generator( n_num = 10 ):
	selecting_number_1 = 0
	selecting_number_2 = 0
	selecting_number_3 = 0
	
	
	seed_1 = tf.random.set_seed(int( datetime.now().microsecond ));
	selecting_number_1 = tf.random.normal([1], 0, 1, dtype=tf.float32, seed=seed_1).numpy()[0]
	
	seed_1 = tf.random.set_seed(int( datetime.now().microsecond ));
	selecting_number_2 = tf.random.normal([1], 0, 1, dtype=tf.float32, seed=seed_1).numpy()[0]
	
	seed_1 = tf.random.set_seed(int( datetime.now().microsecond ));
	selecting_number_3 = tf.random.normal([1], 0, 1, dtype=tf.float32, seed=seed_1).numpy()[0]
	
	series_1 = []
	
	for i in range( n_num ):
		series_1.append( i * selecting_number_1 * selecting_number_2 + selecting_number_3 )
	
	series_1 = tf.keras.layers.Softmax()(series_1)
	
	return series_1.numpy()
```

#### Function call and variables ####

```
series_1 = function_serie_generator( 15 )
series_2 = function_serie_generator( 15 )
series_3 = function_serie_generator( 15 )
series_4 = function_serie_generator( 15 )
series_values = [ series_1, series_2, series_3, series_4 ]
```

## Input ##

![Alt text](https://github.com/jkaewprateep/sequence_classification_deep_learning/blob/main/Figure_21.png "input")

## Results ##

![Alt text](https://github.com/jkaewprateep/sequence_classification_deep_learning/blob/main/Figure_22.png "result")

| File name | Description |
--- | --- |
| sample.py | sample codes |
| Figure_21.png | input series 1 to 4 |
| Figure_22.png | input series 1 to 4 and prediction result |
| README.md | readme file |

## Reference ##

I found this example interesting because there is nowhere without the nature of the something background.

1. https://www.mathworks.com/help/deeplearning/ug/classify-sequence-data-using-lstm-networks.html
