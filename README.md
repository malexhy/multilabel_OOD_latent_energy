## Multi-Label Out-of-Distribution Detection via Latent Energy Attention

This is a PyTorch implementation of Multi-Label Out-of-Distribution Detection via Latent Energy Attention.

## Used libary
* matplotlib 3.10.8
* numpy 1.26.4
* pytorch 2.6.0
* scipy 1.17.1
* torchvision 0.21.0
* torcheval 0.0.7
* tqdm 4.67.3

## Training
Train LAC Model. Available training_set include 'voc', 'coco', 'nus'
```
python main.py --save_dir './save/weight_1' --cla_module 'at_u5__de1' --training_set 'voc'
```
### Limited dataset
Train LAC Model with limited subset by setting the "--training_set". \
Allow dataset: 
* PASCAL-VOC: voc_\* (Replaece \* by present of dataset, e.g. voc_10 = 10% of Pascal_voc)
* NUS-WIDE: nus_\*p or nus_\*n (nus_10p = 10% of NUS-WIDE, nus_500n = 500 NUS-WIDE data)

For example: 
```
python main.py --save_dir './save/weight_1' --cla_module 'at_u5__de1' --training_set 'voc_10'
```

### Training data pre-shuffle
We pre-shuffle the training set to control the label distribution when using limited subset.\
We provide 3 balanced shuffle and 3 imbalanced shuffle.\
Balanced shuffle: 'train_ind0', 'train_ind1', 'train_ind2'\
Imbalanced shuffle: None, 'train_ind_4_0', 'train_ind_4_2'

For example: 
```
python main.py --save_dir './save/weight_1' --cla_module 'at_u5__de1' --training_set 'voc_10' --train_data_ind 'train_ind_4_0'
```

## Eval
Evaluate the trained model.
```
python eval.py --eval_path './save/weight_1'
```