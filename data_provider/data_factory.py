from data_provider.data_loader import KAFASATAnomalyDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'KAFASATAnomalyDataset': KAFASATAnomalyDataset
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    
    # 플래그에 따라 다른 데이터 경로 사용
    if flag == 'train' and hasattr(args, 'train_data_path') and args.train_data_path:
        data_path = args.train_data_path
    elif flag == 'test' and hasattr(args, 'test_data_path') and args.test_data_path:
        data_path = args.test_data_path
    elif flag == 'val' and hasattr(args, 'val_data_path') and args.val_data_path:
        data_path = args.val_data_path
    else:
        data_path = args.data_path

    # target_feature 매개변수 확인 (다변량 모드에서는 None)
    if hasattr(args, 'target_feature') and args.target_feature is not None:
        target_feature = args.target_feature
    else:
        target_feature = None

    # 셔플 설정 등은 동일하게 유지
    if flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    # samples_per_file 매개변수 확인
    samples_per_file = getattr(args, 'samples_per_file', None)
    stride = getattr(args, 'stride', None)
    
    # 데이터셋 생성 시 samples_per_file 추가
    if flag in ['train', 'val']:
        # SimpleTimeSeriesDatasetBenchmark인 경우 samples_per_file 파라미터 전달
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
            
            # 데이터셋에서 feature 정보를 args에 설정
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
                ddp=getattr(args, 'ddp', False),
                stride = stride
            )
            
            # 데이터셋에서 feature 정보를 args에 설정
            if hasattr(data_set, 'feature_names'):
                args.feature_names = data_set.feature_names
            if hasattr(data_set, 'n_var'):
                args.enc_in = data_set.n_var
                args.c_out = data_set.n_var
    else:
        # SimpleTimeSeriesDatasetBenchmark인 경우, 테스트 시에는 전체 데이터 사용 (samples_per_file=None)
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
                samples_per_file=None  # 테스트 시에는 전체 데이터 사용
            )
            
            # 데이터셋에서 feature 정보를 args에 설정
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
            
            # 데이터셋에서 feature 정보를 args에 설정
            if hasattr(data_set, 'feature_names'):
                args.feature_names = data_set.feature_names
            if hasattr(data_set, 'n_var'):
                args.enc_in = data_set.n_var
                args.c_out = data_set.n_var
    print(flag, len(data_set))
    if args.ddp:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last
        )
    return data_set, data_loader
