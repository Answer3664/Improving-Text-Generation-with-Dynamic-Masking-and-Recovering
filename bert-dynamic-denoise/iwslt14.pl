#!/usr/bin/perl

use strict;
use warnings;

#use Cwd;

#my $cur_dir = getcwd;
my $t2t_dir = "/home/lzd/transmodel/bert-dynamic-denoise";
my $workspace = "/home/lzd/mt-exp/iwslt14-de2en/bert-denoise";

my $gpu_id = 1;
my $alpha = 0.2;

my $python3 = "/home/lzd/anaconda3/bin/python3.6";
$python3 = "python3";
my $python2 = "/usr/bin/pythor2.7";
my $bpe_dir = "/home/lzd/open/subword-nmt/subword_nmt/";

my $plain2sgm = "/home/lzd/script/plain2sgm";
my $mteval_11b = "/home/lzd/script/mteval-v11b.pl";
my $multi_bleu = "/home/lzd/script/multi-bleu.perl";
my $fast_score = "/home/lzd/script/fast_score";
my $chrF = "/home/lzd/script/chrF++.py";

my $src = "de";
my $tgt = "en";

my $step0 = 0; #move data
my $step1 = 0; #prepare_data
my $step2 = 0; #training
my $step3 = 1; #tuning
my $step4 = 0; #test
my $step5 = 0; #infer with ensemble model
my $step6 = 0; #test multi model

my $data_dir = "$workspace/data";
makedir("$data_dir");
my $model_dir = "$workspace/model$alpha-m3";
makedir("$model_dir");
my $prep = "$workspace/iwslt14";
makedir("$prep");
my $train_test_data_dir = "$prep";
my $tmp_dir = "$workspace/tmp";
makedir("$tmp_dir");
my $valid = "valid";
my @test_sets = ("test");

my $cmd;

my $data_from = "/home/lzd/corpus/iwslt14/iwslt14.tokenized.de-en";

if ($step0) {
  for my $set ("train") {
    $cmd = "ln -s $data_from/$set.$src $tmp_dir/; ln -s $data_from/$set.$tgt $tmp_dir/";
    run("$cmd");
    $cmd = "ln -s $tmp_dir/$set.$src $prep/$set.$src";
    run("$cmd");
    $cmd = "ln -s $tmp_dir/$set.$tgt $prep/$set.$tgt";
    run("$cmd");

  }
  for my $set ($valid) {
    $cmd = "ln -s $data_from/$set.$src $tmp_dir/; ln -s $data_from/$set.$tgt $tmp_dir/; ln -s $data_from/tmp/$set.$tgt $tmp_dir/$set.tok.$tgt";
    run("$cmd");
    $cmd = "cp $tmp_dir/$set.$src $prep/valid.$src";
    run("$cmd");
    $cmd = "cp $tmp_dir/$set.$tgt $prep/valid.$tgt";
    run("$cmd");
    $cmd = "cp $tmp_dir/$set.tok.$tgt $prep/valid.tok.$tgt";
    run("$cmd");


  }

  for my $set (@test_sets) {
    $cmd = "ln -s $data_from/$set.$src $tmp_dir/; ln -s $data_from/$set.$tgt $tmp_dir/; ln -s $data_from/tmp/$set.$tgt $tmp_dir/$set.tok.$tgt";
    run("$cmd");
    #for my $i ("0", "1", "2", "3") {
    #  $cmd = "ln -s $data_from/$set.tok.$tgt$i $tmp_dir/$set.$tgt$i";
    #  run("$cmd");
    #  $cmd = "ln -s $tmp_dir/$set.$tgt$i $prep/";
    #  run("$cmd");
    #}
    #$cmd = "ln -s $data_from/$set.$tgt $tmp_dir/$set.$tgt";
    #run("$cmd");
    #$cmd = "ln -s $tmp_dir/$set.$tgt $prep/";
    #run("$cmd");
    #$cmd = "ln -s $tmp_dir/$set.$src $prep/$set.$src";
    #run("$cmd");
    $cmd = "cp $tmp_dir/$set.$src $prep/test.$src";
    run("$cmd");
    $cmd = "cp $tmp_dir/$set.$tgt $prep/test.$tgt";
    run("$cmd");
    $cmd = "cp $tmp_dir/$set.tok.$tgt $prep/test.tok.$tgt";
    run("$cmd");

  }
}

if ($step1) {
  die "FILE NOT FOUND" unless (-e "$train_test_data_dir/train.$src" and -e "$train_test_data_dir/train.$tgt" and -e "$train_test_data_dir/valid.$src" and -e "$train_test_data_dir/valid.$tgt");
  $cmd = "$python3 $t2t_dir/preprocess.py -train_src $train_test_data_dir/train.$src -train_tgt $train_test_data_dir/train.$tgt -valid_src $train_test_data_dir/valid.$src -valid_tgt $train_test_data_dir/valid.$tgt -save_data $data_dir/iwslt -share_vocab -src_vocab_size 40000 -tgt_vocab_size 40000 -src_seq_length 1500 -tgt_seq_length 1500 1>$workspace/prepare.log 2>&1";
  run("$cmd");
}

if ($step2) {
  my $world_size = 1;
  my $gpu_ranks = "0";
  my @gpus = split(/,/, $gpu_id);
  $world_size = scalar(@gpus);
  for (my $i = 1; $i < scalar(@gpus); $i++) {
    $gpu_ranks = $gpu_ranks . " $i";
  }
  my $accum_count = 2;
  my $batch_size = 4096;
  my $transformer_ff = 2048;
  my $drop_out = 0.3;
  my $heads = 8;
  my $select_layer = 3; #[0,1,2,3,4,5]
  my $bert_dec_layers = 2;
  $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id $python3 $t2t_dir/train.py -alpha $alpha -bert_dec_layers $bert_dec_layers -select_layer $select_layer -data $data_dir/iwslt -save_model $model_dir/model  -world_size $world_size -gpu_ranks $gpu_ranks -accum_count $accum_count -save_checkpoint_steps 5000 -valid_steps 5000 -report_every 20 -transformer_ff $transformer_ff -heads $heads -keep_checkpoint 40 -seed 3435 -train_steps 250000 -warmup_steps 8000 --share_embeddings --share_decoder_embeddings --position_encoding --optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -learning_rate 1.0 -max_grad_norm 0.0 -batch_size $batch_size -batch_type tokens -normalization tokens -dropout $drop_out -label_smoothing 0.1 -max_generator_batches 128 -param_init 0.0 -param_init_glorot -valid_batch_size 8 1> $model_dir/train.log 2>&1";
  run("$cmd");
}
my @data_set = ("test");

if ($step3) {
  @data_set = ("test");
  foreach my $set (@data_set) {
    my $start_iter = 5000;
    my $end_iter = 310000;
    my $iter = $start_iter;
    while ($iter < $end_iter) {
      if (-e "$model_dir/model_step_$iter.pt" and !-e "$model_dir/$iter/$set.tran") {
        print("Decoding using $model_dir/model_step_$iter.pt\n");
        makedir("$model_dir/$iter");
        run("touch $model_dir/$iter/$set.tran");
        $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id $python3 $t2t_dir/translate.py -batch_size 50 -model $model_dir/model_step_$iter.pt -src $train_test_data_dir/$set.$src -output $model_dir/$iter/$set.tran -beam_size 5 -share_vocab -gpu 1 1>$model_dir/$iter/$set.log 2>&1";
        run("$cmd");

        $cmd = "sed -i \"s/@@ //g\" $model_dir/$iter/$set.tran";
        run("$cmd");
        eval_multibleu("$model_dir/$iter/$set.tran", "$tmp_dir/$set.tok.$tgt");
        #eval_fastscore("$model_dir/$iter/$set.tran", "$tmp_dir/$set.$tgt");
	#eval_chrf("$model_dir/$iter/$set.tran", "$tmp_dir/$set.$tgt");
      }
      $iter += 5000;
    }
  }
}

if ($step4) {
  my $best_iteration = "245000";
  foreach my $set (@test_sets) {
    my $iter = $best_iteration;
    makedir("$model_dir/$iter");
    $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id $python3 $t2t_dir/translate.py -model $model_dir/model_step_$iter.pt -src $train_test_data_dir/$set.$src -output $model_dir/$iter/$set.tran -beam_size 5 -share_vocab -gpu 1 1>$model_dir/$iter/$set.log 2>&1";
    run("$cmd");

    $cmd = "sed -i \"s/@@ //g\" $model_dir/$iter/$set.tran";
    run("$cmd");
    eval_multibleu("$model_dir/$iter/$set.tran", "$train_test_data_dir/$set.tok.$tgt");
    #eval_fastscore("$model_dir/$iter/$set.tran", "$tmp_dir/$set.$tgt");
  }
}

if ($step6) {
  foreach my $set (@test_sets) {
    my $start_iter = 150000;
    my $end_iter = 250000;
    my $iter = $start_iter;
    while ($iter <= $end_iter){
      makedir("$model_dir/$iter");
      $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id $python3 $t2t_dir/translate.py -model $model_dir/model_step_$iter.pt -src $train_test_data_dir/$set.$src -output $model_dir/$iter/$set.tran -beam_size 5 -share_vocab -gpu 1 1>$model_dir/$iter/$set.log 2>&1";
      run("$cmd");

      $cmd = "sed -i \"s/@@ //g\" $model_dir/$iter/$set.tran";
      run("$cmd");
      eval_multibleu("$model_dir/$iter/$set.tran", "$train_test_data_dir/$set.tok.$tgt");
      $iter += 5000;
    }
  }
}


if ($step5) {
  $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id $python3 $t2t_dir/avg_checkpoints.py $model_dir $model_dir/ensemble.pt 20";
  run("$cmd");

  my $best_iteration = "ensemble";
  foreach my $set (@data_set) {
    my $iter = $best_iteration;
    makedir("$model_dir/$iter");
    $cmd = "CUDA_VISIBLE_DEVICES=$gpu_id $python3 $t2t_dir/translate.py -model $model_dir/$iter.pt -src $train_test_data_dir/$set.$src -output $model_dir/$iter/$set.tran -gpu 0 1>$model_dir/$iter/$set.log 2>&1";
    run("$cmd");

    $cmd = "sed -i \"s/@@ //g\" $model_dir/$iter/$set.tran";
    run("$cmd");
    eval_multibleu("$model_dir/$iter/$set.tran", "$tmp_dir/$set.$tgt");
    eval_fastscore("$model_dir/$iter/$set.tran", "$tmp_dir/$set.$tgt");
  }
}

sub eval_fastscore {
  my $tran = shift;
  my $ref = shift;
  my $mycmd = "$fast_score -r $ref -i $tran -m ter 1> $tran.evalter 2>&1";
  run("$mycmd");
}

sub eval_chrf {
  my $tran = shift;
  my $ref = shift;
  my $mycmd = "python $chrF -R $ref -H $tran -s > $tran.chrf";
  run("$mycmd");
}

sub eval_multibleu {
  my $tran = shift;
  my $ref = shift;
  my $mycmd = "perl $multi_bleu $ref < $tran > $tran.evalmb";
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


