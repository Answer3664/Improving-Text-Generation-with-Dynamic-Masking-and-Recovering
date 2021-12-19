#!/usr/bin/perl

use strict;
use warnings;

my $t2t_dir = "/home/zdliu/trans-model/bert-dynamic-self";
my $workspace = "/home/zdliu/amr2text/bert-dynamic";
my $gpu_id = "6";
my $alpha = "0.6";
my $python3 = "/home/zdliu/anaconda3/bin/python3.6";
$python3 = "python3";
my $python2 = "/usr/bin/pythor2.7";
my $bpe_dir = "/home/zdliu/open/subword-nmt/subword_nmt/";

my $plain2sgm = "/home/zdliu/open/plain2sgm";
my $mteval_11b = "/home/zdliu/open/mteval-v11b.pl";
my $multi_bleu = "/home/zdliu/open/multi-bleu.perl";
my $fast_score = "/home/zdliu/open/fast_score";
my $chrF = "/home/zdliu/open/chrF++.py";
my $meteor = "-Xmx2G -jar /home/zdliu/open/meteor-1.5/meteor-1.5.jar";

my $src = "amr";
my $tgt = "txt";
my $step0 = 0; #preprocess data
my $step1 = 0; #train
my $step2 = 1; #translate valid
my $step3 = 0; #translate test

my $data_dir = "$workspace/data";
makedir("$data_dir");
my $model_dir = "$workspace/model-$alpha-0.6-m2";
makedir("$model_dir");

my @test_sets = ("test");
my $cmd;

my $train_test_data_dir = "/home/zdliu/jzhu-data/LDC2017T10/five_path_corpus";

if($step0){
  $cmd = "$python3 $t2t_dir/preprocess.py -train_src $train_test_data_dir/train_concept_no_EOS_bpe -train_tgt $train_test_data_dir/train_target_token_bpe -train_structure1 $train_test_data_dir/train_edge_all_bpe_1 -train_structure2 $train_test_data_dir/train_edge_all_bpe_2 -train_structure3 $train_test_data_dir/train_edge_all_bpe_3 -train_structure4 $train_test_data_dir/train_edge_all_bpe_4 -train_structure5 $train_test_data_dir/train_edge_all_bpe_5 -train_structure6 $train_test_data_dir/train_edge_all_bpe_6 -train_structure7 $train_test_data_dir/train_edge_all_bpe_7 -train_structure8 $train_test_data_dir/train_edge_all_bpe_8 -valid_src $train_test_data_dir/dev_concept_no_EOS_bpe -valid_tgt $train_test_data_dir/dev_target_token_bpe -valid_structure1 $train_test_data_dir/dev_edge_all_bpe_1 -valid_structure2 $train_test_data_dir/dev_edge_all_bpe_2 -valid_structure3 $train_test_data_dir/dev_edge_all_bpe_3 -valid_structure4 $train_test_data_dir/dev_edge_all_bpe_4 -valid_structure5 $train_test_data_dir/dev_edge_all_bpe_5 -valid_structure6 $train_test_data_dir/dev_edge_all_bpe_6 -valid_structure7 $train_test_data_dir/dev_edge_all_bpe_7 -valid_structure8 $train_test_data_dir/dev_edge_all_bpe_8 -save_data $data_dir/gq -src_vocab_size 20000 -tgt_vocab_size 20000 -structure_vocab_size 20000 -src_seq_length 10000 -tgt_seq_length 10000 -share_vocab 1>$workspace/prepare.log 2>&1";
  run("$cmd");
}

if($step1){
  my $world_size = 1;
  my $gpu_ranks = "0";
  my @gpus = split(/,/, $gpu_id);
  $world_size = scalar(@gpus);
  for (my $i = 1; $i < scalar(@gpus); $i++) {
    $gpu_ranks = $gpu_ranks . " $i";
  }
  my $select_layer = 2; #[0,1,2,3,4,5]  
  my $bert_dec_layers = 3;
  $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id nohup $python3 $t2t_dir/train.py -alpha $alpha -data $data_dir/gq -bert_dec_layers $bert_dec_layers -select_layer $select_layer -save_model $model_dir/model -world_size $world_size -gpu_ranks $gpu_ranks -save_checkpoint_steps 5000 -valid_steps 5000 -report_every 20 -keep_checkpoint 30 -seed 3435  -train_steps 300000 -warmup_steps 16000 --share_decoder_embeddings -share_embeddings --position_encoding --optim adam -adam_beta1 0.9 -adam_beta2 0.98 -decay_method noam -learning_rate 0.5 -max_grad_norm 0.0 -batch_size 1024 -batch_type tokens -normalization tokens -dropout 0.3 -label_smoothing 0.1 -max_generator_batches 100 -param_init 0.0 -param_init_glorot -valid_batch_size 4 -accum_count 1  1>> $model_dir/train.log 2>&1 ";
  run("$cmd");

}

my @data_set = ("test");

if($step2){
  @data_set = ("test");
  foreach my $set (@data_set) {
    my $start_iter = 200000;
    my $end_iter = 310000;
    my $iter = $start_iter;
    while ($iter < $end_iter) {
      if (-e "$model_dir/model_step_$iter.pt" and !-e "$model_dir/$iter/$set.tran") {
        print("Decoding using $model_dir/model_step_$iter.pt\n");
        makedir("$model_dir/$iter");
        run("touch $model_dir/$iter/$set.tran");
        $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id $python3 $t2t_dir/translate.py -model $model_dir/model_step_$iter.pt -src $train_test_data_dir/dev_concept_no_EOS_bpe -structure1 $train_test_data_dir/dev_edge_all_bpe_1 -structure2 $train_test_data_dir/dev_edge_all_bpe_2 -structure3 $train_test_data_dir/dev_edge_all_bpe_3 -structure4  $train_test_data_dir/dev_edge_all_bpe_4 -structure5 $train_test_data_dir/dev_edge_all_bpe_5 -structure6 $train_test_data_dir/dev_edge_all_bpe_6 -structure7 $train_test_data_dir/dev_edge_all_bpe_7 -structure8 $train_test_data_dir/dev_edge_all_bpe_8 -output $model_dir/$iter/$set.tran -beam_size 5 -share_vocab -gpu 0 1>$model_dir/$iter/$set.log 2>&1";
        run("$cmd");

        $cmd = "sed -i \"s/@@ //g\" $model_dir/$iter/$set.tran";
        run("$cmd");
        eval_multibleu("$model_dir/$iter/$set.tran", "$train_test_data_dir/dev_target_token");
      }
      #eval_multibleu("$model_dir/$iter/$set.tran", "$train_test_data_dir/dev_target_token");
      #eval_chrf("$model_dir/$iter/$set.tran", "$train_test_data_dir/dev_target_token");
      #eval_meteor("$model_dir/$iter/$set.tran", "$train_test_data_dir/dev_target_token");
      $iter += 5000;
    }
  }
}

if($step3){
  my $best_iteration = "285000";
  foreach my $set (@test_sets) {
    my $iter = $best_iteration;
    makedir("$model_dir/$iter");
    $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id $python3 $t2t_dir/translate.py -model $model_dir/model_step_$iter.pt -src $train_test_data_dir/test_concept_no_EOS_bpe -structure1 $train_test_data_dir/test_edge_all_bpe_1 -structure2 $train_test_data_dir/test_edge_all_bpe_2 -structure3 $train_test_data_dir/test_edge_all_bpe_3 -structure4 $train_test_data_dir/test_edge_all_bpe_4 -structure5 $train_test_data_dir/test_edge_all_bpe_5 -structure6 $train_test_data_dir/test_edge_all_bpe_6 -structure7 $train_test_data_dir/test_edge_all_bpe_7 -structure8 $train_test_data_dir/test_edge_all_bpe_8 -output $model_dir/$set.tran -beam_size 5 -share_vocab -gpu 0 1>$model_dir/$set.log 2>&1";
    run("$cmd");

    $cmd = "sed -i \"s/@@ //g\" $model_dir/$set.tran";
    run("$cmd");
    eval_multibleu("$model_dir/$set.tran", "$train_test_data_dir/test_target_token");
    eval_chrf("$model_dir/$set.tran", "$train_test_data_dir/test_target_token");
    eval_meteor("$model_dir/$set.tran", "$train_test_data_dir/test_target_token");
  }
}


sub eval_multibleu {
  my $tran = shift;
  my $ref = shift;
  my $mycmd = "perl $multi_bleu $ref < $tran > $tran.evalmb";
  run("$mycmd");
}

sub eval_chrf {
  my $tran = shift;
  my $ref = shift;
  my $mycmd = "python $chrF -R $ref -H $tran > $tran.chrf";
  run("$mycmd");
}

sub eval_meteor {
  my $tran = shift;
  my $ref = shift;
  my $mycmd = "java $meteor $tran $ref -norm > $tran.meteor";
  run("$mycmd");
}


sub makedir {
  my $dir = shift;
  my $mycmd = "mkdir -p $dir";
  run("$mycmd") unless -e "$dir";
}

sub run {
  my $mycmd = shift;
  print "$mycmd\n";
  system("$mycmd");
}




