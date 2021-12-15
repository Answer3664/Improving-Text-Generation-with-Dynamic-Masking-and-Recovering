# Text-Genreation-with-DM-and-MLM
The repository contains the code for our paper ["Improving Text Generation with Dynamic Masking and Recovering"](https://www.ijcai.org/proceedings/2021/534) in IJCAI-2021

## Requirements
For NMT and AMR-to-Text generation:

* Python version = 3.6.10
* Pytorch version = 1.1.0

For Image Caption: 
* Python version = 2.7.16
* Pytorch version = 1.1.0

## Datasets

Because the amount of data pre-processed data is very large, for AMR-to-Text generation and Image Caption tasks, we give repositories to the data processing that we use.
* [WMT14EN-DE](https://drive.google.com/file/d/1r_Bv6_a8ylGVOkT1MaRmEaQ4OHylIVOA/view?usp=sharing)
* [IWSLT14DE-EN](https://drive.google.com/file/d/1hotynRPVR1anNdjuThOOLF6gg_B1yWLX/view?usp=sharing)
* [AMR-to-Text](https://github.com/Amazing-J/structural-transformer)
* [Image Caption](https://github.com/ruotianluo/ImageCaptioning.pytorch)


## Run code

For NMT, first, go into the **bert-dynamic-denoise** directory.
Before run with *wmt14en-de.pl* and *iwslt14.pl*, you need to modify the working directory inside.
 ```perl
    #!bin/perl
    my $cur_dir = "/home/lzd/mt-exp/wmt14-en2de";
    my $workspace = "$cur_dir/bert-dynamic-ende";
    my $gpu_id = "1";
    my $python3 = "/home/lzd/anaconda3/bin/python3";
    my $t2t_dir = "/home/lzd/transmodel/bert-dynamic-denoise";
    my $multi_bleu = "/home/lzd/script/multi-bleu.perl";
    ```

### Pre-processing, Training and Validation
You need to set step0-step3 as 1, step4-step5 as 0, such as:
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
You need to set step0-step3 as 0, step4 as 1 such as:
 ```perl
    #!bin/perl
    my $step0 = 0; #move data
    my $step1 = 0; #prepare_data
    my $step2 = 0; #training
    my $step3 = 0; #tuning
    my $step4 = 1; #test
    my $step5 = 0; #infer with ensemble model
    ```
Then you can get the final result.

Also note that, for results in our paper, we do not perform step5.

For AMR-to-Text, go into the **bert-dynamic-self** directory, then follow the same steps as NMT

