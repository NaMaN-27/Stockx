import pandas as pd
import numpy as np

def conditional_prob_calc (weekly_data):
    ranges = [-np.inf, -4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf]

    labels = ["<-4%", "-4% to -3%", "-3% to -2%", "-2% to -1%", "-1% to 0%", 
            "0% to 1%", "1% to 2%", "2% to 3%", "3% to 4%", ">4%"]

    weekly_data['PercentChangePrev'] = weekly_data['PercentChange'].shift(1)

    weekly_data["PrevBin"] = pd.cut(weekly_data["PercentChangePrev"], bins=ranges, labels=labels)

    weekly_data["CurrentBin"] = pd.cut(weekly_data["PercentChange"], bins=ranges, labels=labels)

    prob_df = pd.DataFrame(index=labels, columns=labels)
    count_df = pd.DataFrame(index=labels, columns=labels)

    # Nested loop to calc probs for each combination of PrevBin and CurrentBin
    for prev_bin in labels:
        for current_bin in labels:
            num_joint = len(weekly_data[(weekly_data["PrevBin"] == prev_bin) & 
                                        (weekly_data["CurrentBin"] == current_bin)])
            num_prev = len(weekly_data[weekly_data["PrevBin"] == prev_bin])
            count_df.loc[prev_bin, current_bin] = num_joint
            prob = num_joint / num_prev if num_prev > 0 else 0
            prob_df.loc[prev_bin, current_bin] = prob

    count_df = count_df.astype(int)
    prob_df = prob_df.astype(float)

    # Negative return bins include percentage changes less than or equal to 0%
    negative_return = ["<-4%", "-4% to -3%", "-3% to -2%", "-2% to -1%", "-1% to 0%"]

    # Positive return bins include percentage changes greater than 0%
    positive_return = ["0% to 1%", "1% to 2%", "2% to 3%", "3% to 4%", ">4%"]

    prob_df["Negative"] = prob_df[negative_return].sum(axis=1)
    prob_df["Positive"] = prob_df[positive_return].sum(axis=1)

    count_df["Negative"] = count_df[negative_return].sum(axis=1)
    count_df["Positive"] = count_df[positive_return].sum(axis=1)

    # Calculate the probability of >1%
    prob_df[">1%"] = prob_df["1% to 2%"] + prob_df["2% to 3%"] + prob_df["3% to 4%"] + prob_df[">4%"]

    # Calculate the probability of >2%
    prob_df[">2%"] = prob_df["2% to 3%"] + prob_df["3% to 4%"] + prob_df[">4%"]

    # Calculate the probability of >3%
    prob_df[">3%"] = prob_df["3% to 4%"] + prob_df[">4%"]

    # Calculate the total probability by summing "Positive" and "Negative" columns
    prob_df["Total"] = prob_df["Positive"] + prob_df["Negative"]

    # Calculate the total count by summing "Positive" and "Negative" columns
    count_df["Total"] = count_df["Positive"] + count_df["Negative"]
    
    #Renaming AXES
    prob_df.rename_axis(index="Previous Week", columns="This Week", inplace=True)

    return prob_df, count_df