from data_provider.data_loader import KAFASATAnomalyDataset
from torch.utils.data import DataLoader

data_dict = {
    'KAFASATAnomalyDataset': KAFASATAnomalyDataset
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    
    if flag == 'train' and hasattr(args, 'train_data_path') and args.train_data_path:
        data_path = args.train_data_path
    elif flag == 'test' and hasattr(args, 'test_data_path') and args.test_data_path:
        data_path = args.test_data_path
    elif flag == 'val' and hasattr(args, 'val_data_path') and args.val_data_path:
        data_path = args.val_data_path
    else:
        data_path = args.data_path

    if hasattr(args, 'target_feature') and args.target_feature is not None:
        target_feature = args.target_feature
    else:
        target_feature = None

    if flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    samples_per_file = getattr(args, 'samples_per_file', None)
    stride = getattr(args, 'stride', None)
    
    if flag in ['train', 'val']:
        if args.data == 'SimpleTimeSeriesDatasetBenchmark':
            data_set = Data(
                root_path=args.root_path,
                data_path=data_path,
                flag=flag,
                size=[args.seq_len, args.input_token_len, args.output_token_len],
                nonautoregressive=args.nonautoregressive,
                test_flag=args.test_flag,
                subset_rand_ratio=args.subset_rand_ratio,
                use_full_data=True,
                target_feature=target_feature,
                samples_per_file=samples_per_file,
                stride = stride
            )
            
            if hasattr(data_set, 'feature_names'):
                args.feature_names = data_set.feature_names
            if hasattr(data_set, 'n_var'):
                args.enc_in = data_set.n_var
                args.c_out = data_set.n_var
        else:
            data_set = Data(
                root_path=args.root_path,
                data_path=data_path,
                flag=flag,
                size=[args.seq_len, args.input_token_len, args.output_token_len],
                nonautoregressive=args.nonautoregressive,
                test_flag=args.test_flag,
                subset_rand_ratio=args.subset_rand_ratio,
                use_full_data=True,
                target_feature=target_feature,
                stride = stride
            )
            
            if hasattr(data_set, 'feature_names'):
                args.feature_names = data_set.feature_names
            if hasattr(data_set, 'n_var'):
                args.enc_in = data_set.n_var
                args.c_out = data_set.n_var
    else:
        if args.data == 'SimpleTimeSeriesDatasetBenchmark':
            data_set = Data(
                root_path=args.root_path,
                data_path=data_path,
                flag=flag,
                size=[args.test_seq_len, args.input_token_len, args.test_pred_len],
                nonautoregressive=args.nonautoregressive,
                test_flag=args.test_flag,
                subset_rand_ratio=args.subset_rand_ratio,
                use_full_data=True,
                target_feature=target_feature,
                samples_per_file=None
            )
            
            if hasattr(data_set, 'feature_names'):
                args.feature_names = data_set.feature_names
            if hasattr(data_set, 'n_var'):
                args.enc_in = data_set.n_var
                args.c_out = data_set.n_var
        else:
            data_set = Data(
                root_path=args.root_path,
                data_path=data_path,
                flag=flag,
                size=[args.test_seq_len, args.input_token_len, args.test_pred_len],
                nonautoregressive=args.nonautoregressive,
                test_flag=args.test_flag,
                subset_rand_ratio=args.subset_rand_ratio,
                use_full_data=True,
                target_feature=target_feature
            )
            
            if hasattr(data_set, 'feature_names'):
                args.feature_names = data_set.feature_names
            if hasattr(data_set, 'n_var'):
                args.enc_in = data_set.n_var
                args.c_out = data_set.n_var
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=drop_last,
    )
    return data_set, data_loader
