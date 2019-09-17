ph_setting1 = {"filter_sizes": "2,3,4", "num_filters": 200,
            "dropout_keep_prob": 0.8, "l2_reg_lambda": 0.001,
            "batch_size": 64, "num_epochs": 100,
            "evaluate_every": 20, "allow_soft_placement": True, "log_device_placement": False}

ph_setting2 = {"filter_sizes": "2,3,4", "num_filters": 200,
            "dropout_keep_prob": 0.8, "l2_reg_lambda": 0.001,
            "batch_size": 64, "num_epochs": 100,
            "evaluate_every": 20, "allow_soft_placement": True, "log_device_placement": False}

ph_setting3 = {"filter_sizes": "2,3,4", "num_filters": 200,
            "dropout_keep_prob": 0.8, "l2_reg_lambda": 0.001,
            "batch_size": 64, "num_epochs": 100,
            "evaluate_every": 20, "allow_soft_placement": True, "log_device_placement": False}

ph_setting4 = {"filter_sizes": "2,3,4", "num_filters": 200,
            "dropout_keep_prob": 0.8, "l2_reg_lambda": 0.001,
            "batch_size": 64, "num_epochs": 100,
            "evaluate_every": 20, "allow_soft_placement": True, "log_device_placement": False}

ph_setting5 = {"filter_sizes": "2,3,4", "num_filters": 200,
            "dropout_keep_prob": 0.8, "l2_reg_lambda": 0.001,
            "batch_size": 64, "num_epochs": 100,
            "evaluate_every": 20, "allow_soft_placement": True, "log_device_placement": False}

ph_model_settings = {"dataset_path": "", "vocab_path": "",
                     "vector_path": "", "ms_path": "", "oov_mode": "pure", "cv_k": 5, "test_path": 'test/ph_test.pickle',
                     "lang": 'ph', "window_size": 5, "embedding_dim": 200, 'checkpoint_dir': "logs/cnn/records/ph",
                     "setting": [ph_setting1, ph_setting2, ph_setting3, ph_setting4, ph_setting5]}

gh_model_settings = {"dataset_path": "", "vocab_path": "",
                     "vector_path": "", "ms_path": "", "oov_mode": "pure", "cv_k": 5, "test_path": 'test/gh_test.pickle',
                     "lang": 'gh', "window_size": 5, "embedding_dim": 200, 'checkpoint_dir': "logs/cnn/records/gh",
                     "setting": [ph_setting1, ph_setting2, ph_setting3, ph_setting4, ph_setting5]}

vz_setting1 = {"filter_sizes": "1,2,3", "num_filters": 200,
            "dropout_keep_prob": 0.8, "l2_reg_lambda": 0.001,
            "batch_size": 64, "num_epochs": 100,
            "evaluate_every": 20, "allow_soft_placement": True, "log_device_placement": False}

vz_setting2 = {"filter_sizes": "1,2,3", "num_filters": 200,
            "dropout_keep_prob": 0.8, "l2_reg_lambda": 0.001,
            "batch_size": 64, "num_epochs": 100,
            "evaluate_every": 20, "allow_soft_placement": True, "log_device_placement": False}

vz_setting3 = {"filter_sizes": "1,2,3", "num_filters": 200,
            "dropout_keep_prob": 0.8, "l2_reg_lambda": 0.001,
            "batch_size": 64, "num_epochs": 100,
            "evaluate_every": 20, "allow_soft_placement": True, "log_device_placement": False}

vz_setting4 = {"filter_sizes": "1,2,3", "num_filters": 200,
            "dropout_keep_prob": 0.8, "l2_reg_lambda": 0.001,
            "batch_size": 64, "num_epochs": 100,
            "evaluate_every": 20, "allow_soft_placement": True, "log_device_placement": False}

vz_setting5 = {"filter_sizes": "1,2,3", "num_filters": 200,
            "dropout_keep_prob": 0.8, "l2_reg_lambda": 0.001,
            "batch_size": 64, "num_epochs": 100,
            "evaluate_every": 20, "allow_soft_placement": True, "log_device_placement": False}

vz_model_settings = {"dataset_path": "", "vocab_path": "",
                     "vector_path": "", "ms_path": "", "oov_mode": "pure", "cv_k": 5, "test_path": 'test/vz_test.pickle',
                     "lang": 'vz', "window_size": 5, "embedding_dim": 200, 'checkpoint_dir': "logs/cnn/records/vz",
                     "setting": [vz_setting1, vz_setting2, vz_setting3, vz_setting4, vz_setting5]}
