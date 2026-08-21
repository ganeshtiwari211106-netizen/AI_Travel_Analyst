import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_cleaning import load_and_clean_data

def generate_visualizations(df, output_dir="visualizations"):
    # 1. Price Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Price"], kde=True, bins=30, color="blue")
    plt.title("Distribution of Flight Prices")
    plt.savefig(f"{output_dir}/viz1_price_dist.png")
    plt.close()

    # 2. Days Before Departure vs Price
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df, x="Days_Before_Departure", y="Price", color="red")
    plt.title("Impact of Booking Window on Price")
    plt.savefig(f"{output_dir}/viz2_days_vs_price.png")
    plt.close()

    # 3. Airline vs Price
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Airline", y="Price")
    plt.xticks(rotation=45)
    plt.title("Price Variation by Airline")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/viz3_airline_price.png")
    plt.close()

    # 4. Stops vs Price
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="Total_Stops", y="Price")
    plt.title("Average Price by Number of Stops")
    plt.savefig(f"{output_dir}/viz4_stops_price.png")
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(6, 5))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/viz5_heatmap.png")
    plt.close()
    print("5 Visualizations successfully saved to the visualizations/ folder.")

if __name__ == "__main__":
    df = load_and_clean_data()
    generate_visualizations(df)