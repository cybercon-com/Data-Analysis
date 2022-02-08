import numpy as np               # for pandas
import pandas as pd              # DataFrames
import matplotlib.pyplot as plt  # plot graphs
import xlrd                      # read excel documents
from scipy import stats          # linear regression analysis

## display DataFrames in full
pd.set_option("display.max_rows", None, "display.max_columns", None)

def HappinessFunction(year,     # of type 'integer'
                      file_name # of type 'string'
                      ):
    happiness_df = pd.read_excel(file_name)
    if year == 2021 or year == 2020:
        happiness_df = happiness_df[["Country name", "Ladder score"]]
        happiness_df.columns = ["Country", "Happiness Rating"]
        happiness_df["Year"] = [year for i in range(len(happiness_df))]
    else:
        happiness_df = happiness_df[["Country name", "Year", "Life Ladder"]]
        happiness_df.columns = ["Country", "Year", "Happiness Rating"]
    #print(happiness_df)
    return happiness_df

happiness_df = pd.DataFrame()
pearson_correlations_dict = {}
def ConsumableFunction(consumables,                        # either of type 'list' or 'string'
                       units, people, time,                # all of type 'string'
                       file_path,                          # of type 'string'
                       graph = False,                      # of type 'Boolean', default value = False
                       linear_regression_analysis = False, # of type 'Boolean', default value = False
                       use_scipy = False                   # of type 'Boolean', default value = False
                       ):
    consumable_df = pd.read_csv(file_path)
    # delete continent data
    consumable_df.dropna(subset = ["Code"], inplace = True)
    consumable_df = consumable_df.drop(["Code"], axis = 1)
    if type(consumables) == list:
        for consumable in consumables:
            consumable = consumable.title()
        column_names = ["Country", "Year"]
        column_names.extend([units.title() + " of " + consumable + " Consumed per " + people.title() + " per " + time.title() \
                             for consumable in consumables])
        consumable_df.columns = column_names
    else:
        column_name = units.title() + " of " + consumables + " Consumed per " + people.title() + " per " + time.title()
        consumable_df.columns = ["Country", "Year", column_name]
    #print(consumable_df)
    consumable_happiness_df = happiness_df.merge(consumable_df, how = "inner")
    #print(consumable_happiness_df)
    pearson_correlation = list(consumable_happiness_df.corr(method = "pearson").iloc[0].iloc[2:])
    #print(pearson_correlation)
    # add pearson correlation to a dictionary
    for i, correlation in enumerate(pearson_correlation, start = 0):
        if type(consumables) == list:
            pearson_correlations_dict[consumables[i]] = correlation
        else:
            pearson_correlations_dict[consumables] = correlation
    consumable_column_names = list(consumable_happiness_df.columns)[3:]
    if linear_regression_analysis == True:
        # only one consumable
        if type(consumables) == str:
            linear_regression_df = consumable_happiness_df[["Happiness Rating", consumable_column_names[0]]].copy()
            linear_regression_df.columns = ["y", "x"]
            if use_scipy == False:
                # y = a + b * x
                # a = y-intercept
                # b = gradient/slope
                linear_regression_df["y * x"] = linear_regression_df["y"] * linear_regression_df["x"]
                linear_regression_df["x ** 2"] = linear_regression_df["x"] ** 2
                linear_regression_df["y ** 2"] = linear_regression_df["y"] ** 2
                number_of_points = len(linear_regression_df)
                b_estimate = (number_of_points * linear_regression_df["y * x"].sum() - linear_regression_df["x"].sum() * \
                              linear_regression_df["y"].sum()) / (number_of_points * linear_regression_df["x ** 2"].sum() - \
                                                                  linear_regression_df["x"].sum() ** 2)
                a_estimate = (linear_regression_df["y"].sum() - b_estimate * linear_regression_df["x"].sum()) / number_of_points
                linear_regression_df["y_estimate"] = a_estimate + b_estimate * linear_regression_df["x"]
                linear_regression_df["error"] = linear_regression_df["y"] - linear_regression_df["y_estimate"]
                residual_sum = linear_regression_df["error"].sum()
                #print("residual sum: " + str(residual_sum))
                # R ** 2 = 1 - RSS / TSS
                # RSS = Residual Sum of Squares
                # TSS = Total Sum of Squares
                RSS = (linear_regression_df["error"] ** 2).sum()
                y_mean = linear_regression_df["y_estimate"].mean()
                TSS = ((linear_regression_df["y_estimate"] - y_mean) ** 2).sum()
                R_squared = 1 - RSS / TSS
                #print("R squared (coefficient of determination): " + str(R_squared))
                # RMSE = Root Mean Squared Error
                RMSE = (linear_regression_df["error"] ** 2 / number_of_points).sum() ** 0.5
                #print("Root Mean Squared Error: " + str(RMSE))
                #print("y-intercept: " + str(a_estimate))
                #print("gradient: " + str(b_estimate))
                #print(linear_regression_df)
            else:
                # y = a + b * x
                # a = y-intercept
                # b = gradient/slope
                x = linear_regression_df["x"].to_list()
                y = linear_regression_df["y"].to_list()
                gradient, intercept, r_value, p_value, standard_error = stats.linregress(x, y)
                def linear_regression_function(x):
                    return gradient * x + intercept
                linear_regression_model = list(map(linear_regression_function, x))
                #print("r value: " + str(r_value))
                #print("standard error: " + str(standard_error))
                #print("y-intercept: " + str(intercept))
                #print("gradient: " + str(gradient))
    if graph == True:
        figure = 0
        for consumable in consumable_column_names:
            plt.figure(figure)
            plt.scatter(consumable_happiness_df[consumable], consumable_happiness_df["Happiness Rating"], marker = "x")
            if linear_regression_analysis == True:
                if use_scipy == False:
                    plt.plot(linear_regression_df["x"], linear_regression_df["y_estimate"], color = "orange")
                else:
                    plt.plot(x, linear_regression_model, color = "orange")
            plt.xlabel(consumable)
            plt.ylabel("Happiness Rating")
            plt.grid()
            plt.show()
            figure += 1

happiness_2021_df = HappinessFunction(2021, 
"C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/Happiness Data/Happiness 2021.xls")
happiness_2020_df = HappinessFunction(2020, 
"C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/Happiness Data/Happiness 2020.xls")
happiness_before_2019_df = HappinessFunction("before 2019", 
"C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/Happiness Data/Happiness Before 2019.xls")

happiness_df = pd.concat([happiness_2021_df, happiness_2020_df, happiness_before_2019_df])
happiness_df = happiness_df.sort_values(by = ["Country", "Year"])
#print(happiness_df)

ConsumableFunction(["Cereals", "Root Tubers", "Vegetables", "Fruits", "Milk \n Products", "Red Meat", "Poultry", "Eggs", "Seafood", \
                    "Legumes", "Nuts", "Oils", "Sugar"],
                   "Grams", "Capita", "Day",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/eat-lancet-diet-comparison.csv")
ConsumableFunction("Chocolate",
                   "Kilograms", "Capita", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/chocolate-consumption-per-person.csv")
ConsumableFunction("Meat",
                   "Kilograms", "Capita", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/daily-meat-consumption-per-person.csv", 
                   graph = True, linear_regression_analysis = True, use_scipy = True)
ConsumableFunction("Bovine",
                   "Kilograms", "Capita", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/Different Meats/beef-and-buffalo-meat-consumption-per-person.csv")
ConsumableFunction("Seafood",
                   "Kilograms", "Capita", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/Different Meats/fish-and-seafood-consumption-per-capita.csv")
ConsumableFunction("Fruits",
                   "Kilograms", "Capita", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/fruit-consumption-per-capita.csv")
ConsumableFunction(["Bananas", "Dates", "Other Citrus \n Fruits", "Oranges \n and \n Mandarins", "Apples", "Lemons \n and Limes", \
                    "Grapes \n (excluding \n wine)", "Grapefruits", "Pineapples", "Plantains", "Other Fruits"],
                   "Kilograms", "Capita", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/fruit-consumption-by-fruit-type.csv")
ConsumableFunction("Alcohol",
                   "Kilograms", "Capita", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/per-capita-alcohol-consumption-kilograms-per-year.csv")
ConsumableFunction("Beer",
                   "Litres", "Capita (Age: 15+)", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/Different Alcohols/beer-consumption-per-person.csv")
ConsumableFunction("Wine",
                   "Litres", "Capita (Age: 15+)", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/Different Alcohols/wine-consumption-per-person.csv")
ConsumableFunction("Spirits",
                   "Litres", "Capita (Age: 15+)", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/Different Alcohols/spirits-consumption-per-person.csv")
ConsumableFunction("Eggs",
                   "Kilograms", "Capita", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/per-capita-egg-consumption-kilograms-per-year.csv")
'''
ConsumableFunction("Electricity",
                   "Kilowatt-hours", "Capita", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/per-capita-electricity-consumption.csv")
'''
ConsumableFunction("Milk",
                   "Kilograms", "Capita", "Year",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/per-capita-milk-consumption.csv")
ConsumableFunction("Cigarettes",
                   "Number", "Adult", "Day",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/sales-of-cigarettes-per-adult-per-day.csv")
ConsumableFunction("Vegetables",
                   "Number", "Adult", "Day",
                   "C:/Users/David Lu/Downloads/What Consumable Makes Us Happiest/vegetable-consumption-per-capita.csv")

pearson_correlations_df = pd.DataFrame.from_dict(pearson_correlations_dict, orient = "index", columns = ["Pearson Correlation"])
pearson_correlations_df = pearson_correlations_df.sort_values(by = "Pearson Correlation", ascending = False)
#print(pearson_correlations_df)
plt.figure(0)
colors = ["magenta" for i in range(len(pearson_correlations_df) - 1)]
colors.insert(0, "purple")
plt.bar(pearson_correlations_df.index, pearson_correlations_df["Pearson Correlation"], align = "center", color = colors)
plt.xticks(fontsize = 10, rotation = 45) # different on different monitors
plt.xlabel("Consumables")
plt.ylabel("Pearson Correlation")
plt.grid()
#plt.show()
