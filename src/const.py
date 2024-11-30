from pathlib import Path

DIR = Path.cwd()
PATH_TEST_DIR = Path(DIR, '../data/test')
PATH_TEST_LABELS = Path( DIR, '../data/test.tsv')
PATH_TRAIN_DIR = Path( DIR, '../data/train')
PATH_TRAIN_LABELS = Path( DIR, '../data/train.tsv')
PREDICT_PATH = Path(DIR, '../data/test')
CHECKPOINTS_PATH = Path(DIR)
FROM_CHECKPOINT_PATH = Path(DIR, '../data/ocr_transformer_4h2l_simple_conv_64x256.pt')
WEIGHTS_PATH = Path(DIR, '../data/ocr_transformer_4h2l_simple_conv_64x256.pt')
PATH_TEST_RESULTS = Path(DIR, '../data/test_rn50_4h2l_result.tsv')
TRAIN_LOG = Path(DIR, '../data/train_log.tsv')
