# Improving-Text-Generation-with-Dynamic-Masking-and-Recovering
The repository contains the code for our paper ["Improving Text Generation with Dynamic Masking and Recovering"](https://www.ijcai.org/proceedings/2021/534) in IJCAI-2021.

## Requirements
For NMT and AMR-to-Text generation:

* Python version = 3.6.10
* Pytorch version = 1.1.0

For Image Caption: 
* Python version = 2.7.16
* Pytorch version = 1.1.0

## Datasets

We upload the datasets we used for NMT task. However, due to the amount of data pre-processed data is very large, for AMR-to-Text generation and Image Caption tasks, we give repositories to the data processing that we use.
* [WMT14EN-DE](https://drive.google.com/file/d/1r_Bv6_a8ylGVOkT1MaRmEaQ4OHylIVOA/view?usp=sharing)
* [IWSLT14DE-EN](https://drive.google.com/file/d/1hotynRPVR1anNdjuThOOLF6gg_B1yWLX/view?usp=sharing)
* [AMR-to-Text](https://github.com/Amazing-J/structural-transformer)
* [Image Caption](https://github.com/ruotianluo/ImageCaptioning.pytorch)


## NMT

For NMT, first, go into the **dynamic-denoise-nmt** directory.
Before run with ```wmt14en-de.pl``` and ```iwslt14.pl```, you need to modify the working directory inside.
 ```perl
    #!bin/perl
    my $cur_dir = "/home/lzd/mt-exp/wmt14-en2de";
    my $workspace = "$cur_dir/bert-dynamic-ende";
    my $gpu_id = "1";
    my $python3 = "/home/lzd/anaconda3/bin/python3";
    my $t2t_dir = "/home/lzd/transmodel/dynamic-denoise-nmt";
    my $multi_bleu = "/home/lzd/script/multi-bleu.perl";
 ```

### Pre-processing, Training and Validation
You need to set ```$step0-$step3``` as 1, ```$step4-$step5``` as 0, such as:
 ```perl
    #!bin/perl
    my $step0 = 1; #move data
    my $step1 = 1; #prepare_data
    my $step2 = 1; #training
    my $step3 = 1; #tuning
    my $step4 = 0; #test
    my $step5 = 0; #infer with ensemble model
 ```

### Decoding
You need to set ```$step0-$step3``` as 0, ```$step4``` as 1 such as:
 ```perl
    #!bin/perl
    my $step0 = 0; #move data
    my $step1 = 0; #prepare_data
    my $step2 = 0; #training
    my $step3 = 0; #tuning
    my $step4 = 1; #test
    my $step5 = 0; #infer with ensemble model
```
Also, you need to specify the ```$best_iteration``` to run testing.
```
if ($step4) {
  my $best_iteration = "290000";
  foreach my $set (@test_sets) {
    my $iter = $best_iteration;
    makedir("$model_dir/$iter");
    $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id $python3 $t2t_dir/translate.py -model $model_dir/model_step_$iter.pt -src $train_test_data_dir/$set.$src -output $model_dir/$iter/$set.tran -beam_size 5 -gpu 1 1>$model_dir/$iter/$set.log 2>&1";
    run("$cmd");

    $cmd = "sed -i \"s/@@ //g\" $model_dir/$iter/$set.tran";
    run("$cmd");
    eval_multibleu("$model_dir/$iter/$set.tran", "$train_test_data_dir/$set.tok.$tgt");
    #eval_fastscore("$model_dir/$iter/$set.tran", "$tmp_dir/$set.$tgt");
  }
} 
```
Then you can get the final results in the directory you specified.

Also note that, for results in our paper, we do not perform ```$step5```.

## AMR
For AMR-to-Text, go into the **dynamic-denoise-amr** directory, then follow the same steps as NMT.


## Cite 

If you like our paper, please cite
```
@inproceedings{ijcai2021-534,
  title     = {Improving Text Generation with Dynamic Masking and Recovering},
  author    = {Liu, Zhidong and Li, Junhui and Zhu, Muhua},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {3878--3884},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/534},
  url       = {https://doi.org/10.24963/ijcai.2021/534},
}
```
