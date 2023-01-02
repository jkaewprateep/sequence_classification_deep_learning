# sequence_classification_deep_learning
For study Sequence Classification Using Deep Learning, it is simply tasks but the prediction accuracy and precisions is on your works. Create functions from random functions as series and noises as experiment input perform initial validated, random function validated by the number of parameters and its input.

## Create inputs from randoms series ##

Our objectives are
1. Determined the series input prediction as training label series correct with the same performance selection by visual graph ( optimizer learning rates = 0.001 ).
2. Create a customized location for custom LSTM input, initial values are created one you start copying declared the layer.
3. It had an integration location where the LSTM layer can train and copy the weights and biases values.

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

![Alt text](https://github.com/jkaewprateep/sequence_classification_deep_learning/blob/main/Figure_25.png "input")

## Results ##

![Alt text](https://github.com/jkaewprateep/sequence_classification_deep_learning/blob/main/Figure_26.png "input")

| File name | Description |
--- | --- |
| sample.py | sample codes |
| Figure_25.png | input series 1 to 4 |
| Figure_26.png | input series 1 to 4 and prediction result |
| README.md | readme file |

## Reference ##

I found this example interesting because there is nowhere without the nature of the something background.

1. https://www.mathworks.com/help/deeplearning/ug/classify-sequence-data-using-lstm-networks.html
