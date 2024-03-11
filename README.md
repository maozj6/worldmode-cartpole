

## Collect data

Please change the render function to get a larger cart pole:
```python
def render(self, mode="human"):
   screen_width = 600
   screen_height = 400


   world_width = self.x_threshold * 2
   scale = screen_width / world_width
   polewidth = 20.0
   polelen = 2*scale * (2 * self.length)
   cartwidth = 60.0
   cartheight = 40.0
```
For training data, each time we collect 30,000; total 3*30,000

```
python collect_train.py --model="CTRL_PATH" --out="SAVE_PATH"
```

For test data, each time we collect 10,000; total 3*10,000
```
python collect_test.py --model="CTRL_PATH" --out="SAVE_PATH"
```



## Train  VAE

Before submitting a task to the GPU node, please change the data path.
```
sbatch train_large.sh
```

## Train  Evaluator

Before submitting a task to the GPU node, please change the data path.
```
sbatch traineva.sh
```

## Train  MDN-LSTM

Before submitting a task to the GPU node, please change the data path.
```
sbatch train_mdn.sh
```


## Test  MDN-LSTM

Before submitting a task to the GPU node, please change the data path.
```
sbatch train_mdn.sh
```
