from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_players_path: str
    raw_games_path: str
    raw_odds_path: str

@dataclass
class DataValidationArtifact:
    validated_players_path: str
    validated_games_path: str
    validated_odds_path: str
    report_file_path: str

@dataclass
class DataTransformationArtifacts:
    tranformed_data_path: str
    split_yaml_path: str

