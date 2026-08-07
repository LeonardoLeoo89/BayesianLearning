import numpy as np
import pandas as pd
import os

def generate_tsunami_sem(n_samples: int = 1000) -> pd.DataFrame:
    """Generates continuous data for the Tsunami Risk SEM."""
    E = np.random.normal(0, 1, n_samples)
    S = np.random.normal(0, 1, n_samples)

    T = 1.8 * S + 0.5 * np.exp(E) + np.random.normal(0, 0.5, n_samples)
    W = 1.2 * T + 0.9 * E + np.random.normal(0, 0.5, n_samples)

    df = pd.DataFrame({
        'Earthquake': E,
        'SubmarineProximity': S,
        'TsunamiHeight': T,
        'WarningUrgency': W
    })
    return df

def generate_allergy_sem(n_samples: int = 1000) -> pd.DataFrame:
    """Generates continuous data for the Allergy Cross-Reactivity SEM."""
    A = np.random.normal(0, 1, n_samples)

    D = 1.2 * A + np.random.normal(0, 0.5, n_samples)
    P = 1.5 * A + np.random.normal(0, 0.5, n_samples)

    B = 1.5 * P + 0.3 * (P**2) + np.random.normal(0, 0.5, n_samples)
    R = 1.1 * P + np.random.normal(0, 0.5, n_samples)

    Ast = 0.5 * (D**2) + 0.5 * D + np.random.normal(0, 0.5, n_samples)

    App = 1.6 * B + np.random.normal(0, 0.5, n_samples)
    Haz = 1.3 * B + np.random.normal(0, 0.5, n_samples)

    df = pd.DataFrame({
        'Atopy': A,
        'DustMiteIgE': D,
        'PollenIgE': P,
        'BirchPollenIgE': B,
        'RhinitisSev': R,
        'AsthmaSev': Ast,
        'AppleIgE': App,
        'HazelnutIgE': Haz
    })
    return df

def generate_train_delay_sem(n_samples: int = 1000) -> pd.DataFrame:
    """Generates continuous data for the Train Delay SEM."""
    Sea = np.random.normal(0, 1, n_samples)
    ToD = np.random.normal(0, 1, n_samples)

    W = 1.5 * Sea + np.random.normal(0, 0.5, n_samples)

    P = 2.0 * ToD + 0.5 * (ToD**2) + np.random.normal(0, 0.5, n_samples)

    TI = 1.2 * W + np.random.normal(0, 0.5, n_samples)
    IF = 1.1 * W + 0.8 * TI + np.random.normal(0, 0.5, n_samples)
    SR = 1.3 * TI + 0.9 * W + np.random.normal(0, 0.5, n_samples)

    HC = 0.5 * (P**2) + np.random.normal(0, 0.5, n_samples)

    DD = 1.2 * IF + 1.1 * HC + 0.5 * (IF * HC) + np.random.normal(0, 0.5, n_samples)

    AD = 1.0 * DD + 1.5 * SR + np.random.normal(0, 0.5, n_samples)
    CC = 2.5 * AD + np.random.normal(0, 0.5, n_samples)

    df = pd.DataFrame({
        'SeasonalFactor': Sea,
        'TimeOfDayRush': ToD,
        'WeatherSev': W,
        'PassengerVol': P,
        'TrackIncident': TI,
        'InfraFailure': IF,
        'SpeedRestriction': SR,
        'HubCongestion': HC,
        'DepartureDelay': DD,
        'ArrivalDelay': AD,
        'CompensationClaim': CC
    })
    return df

if __name__ == "__main__":
    os.makedirs("sem_data", exist_ok=True)

    print("Generating continuous SEM datasets with NONLINEAR dynamics (n=1000)...")

    df_tsunami = generate_tsunami_sem()
    df_tsunami.to_csv("sem_data/tsunami_sem.csv", index=False)
    print(f"Generated Tsunami SEM data with shape: {df_tsunami.shape}")

    df_allergy = generate_allergy_sem()
    df_allergy.to_csv("sem_data/allergy_sem.csv", index=False)
    print(f"Generated Allergy SEM data with shape: {df_allergy.shape}")

    df_train = generate_train_delay_sem()
    df_train.to_csv("sem_data/train_delay_sem.csv", index=False)
    print(f"Generated Train Delay SEM data with shape: {df_train.shape}")

    print("\nDatasets saved to 'sem_data/' directory.")
