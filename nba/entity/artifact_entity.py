from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_players_path: str
    raw_games_path: str

@dataclass
class DataValidationArtifact:
    validated_players_path: str
    validated_games_path: str
    report_file_path: str

