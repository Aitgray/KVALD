import json
from synthetic_data import GlareDataset
from real_data import RealGlareDataset

def get_dataset(config_path: str = 'C:/Users/aidan/Desktop/Projects/KVALD/prototypes/config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)

    source = config.get('data', {}).get('source', 'synthetic')

    if source == 'synthetic':
        return GlareDataset(n_videos=config['train']['n_videos'])
    elif source == 'bdd100k':
        return RealGlareDataset(
            manifest_path=config['data']['manifest_path'],
            glare_config=config['data']['glare'],
            sequence_len=config['data']['sequence_len']
        )
    else:
        raise ValueError(f"Unknown data source: {source}")
