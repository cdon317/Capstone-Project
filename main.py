# import packages:
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as pgo
import scipy.stats
import seaborn as sns
import statsmodels.api as sm

# import CSV as DF:
DF = pd.read_csv(r'C:\Users\cdona\Desktop\World Energy Consumption.csv', usecols=['iso_code', 'country', 'year',
                            'electricity_generation', 'biofuel_electricity', 'coal_electricity',
                            'fossil_electricity', 'gas_electricity', 'hydro_electricity', 'oil_electricity',
                            'renewables_electricity', 'solar_electricity', 'wind_electricity'])

# list of countries for later analysis:
countries = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Angola', 'Antigua and Barbuda', 'Argentina',
             'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados',
             'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina',
             'Botswana', 'Brazil', 'British Virgin Islands', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi',
             'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad',
             'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Cook Islands', 'Costa Rica', 'Cote d\'Ivoire',
             'Croatia', 'Cuba', 'Cyprus', 'Czechia', 'Czechoslovakia', 'Democratic Republic of Congo', 'Denmark',
             'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea',
             'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Faeroe Islands', 'Falkland Islands', 'Fiji', 'Finland',
             'France', 'French Guiana', 'French Polynesia', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana',
             'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 'Guinea',
             'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia',
             'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya',
             'Kiribati', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya',
             'Lithuania', 'Luxembourg', 'Macau', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta',
             'Martinique', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Mongolia', 'Montenegro', 'Montserrat',
             'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'Netherlands Antilles',
             'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'North Korea', 'North Macedonia',
             'Northern Marian Islands', 'Norway', 'Oman', 'Pakistan', 'Palestine', 'Panama', 'Papua New Guinea',
             'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Reunion', 'Romania',
             'Russia', 'Rwanda', 'Saint Helena', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Pierre and Miquelon',
             'Saint Vincent and the Grenadines', 'Samoa', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
             'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia',
             'South Africa', 'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden',
             'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor', 'Togo', 'Tonga',
             'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'USSR', 'Uganda',
             'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'United States Virgin Islands',
             'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Yemen', 'Yugoslavia', 'Zambia', 'Zimbabwe']

# list of usable regions for later analysis (Central America, USSR, CIS, and OPEC removed to avoid duplication of data):
regions = ['Europe', 'North America', 'South & Central America', 'Other Southern Africa', 'Eastern Africa',
           'Asia Pacific', 'Other Northern Africa', 'Middle Africa', 'Western Africa', 'Middle East']

# adjust size of display in interpreter:
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# begin cleaning data; replace all null values with '0':
DF = DF.fillna(method='ffill').fillna(method='bfill')

# convert 'year' column from float dates to integers:
DF['year'] = pd.to_numeric(DF['year'])

# convert 'iso_code' and 'country' both to strings:
DF['iso_code'] = DF['iso_code'].astype(str)
DF['country'] = DF['country'].astype(str)

# check for null values:
check = DF.isnull().sum()

# DF copy organized by countries, used for map plotting:
DF_countries = DF.loc[DF['country'].isin(countries)]
DF_countries = DF_countries[DF_countries.country != 'World']

# DF copy organized by regions, used in all bar charts:
DF_regions = DF.loc[DF['country'].isin(regions)]
DF_regions = DF_regions[DF_regions.country != 'World']

DF_renewable = DF[['iso_code', 'country', 'year', 'solar_electricity', 'wind_electricity', 'biofuel_electricity',
                'hydro_electricity']]

DF_fossil = DF[['iso_code', 'country', 'year', 'coal_electricity', 'gas_electricity', 'oil_electricity']]

def humanize_string(string):
    """Converts underscores from column header to spaces, capitalizes all words."""

    replacement = string.replace('_', ' ')
    replacement = replacement.title()

    return str(replacement)

def plot_map_one_year(col, h):
    """Use Plotly to plot a map, after providing needed values. use_frame = used DF, col = CSV column, ano1 = year
    value. Returns a map."""
    # DATA MOST ACCURATE ON OR AFTER THE YEAR 2000

    # Filtering df to match provided column and year:
    df = DF_countries[['iso_code', 'country', str(col)]].loc[(DF_countries['year'] == int(h))]

    # Finding maximum of provided column to act as y limit in map:
    maximum = df[col].max()

    # If statements to determine text strings to be used in the map headings:
    if str(col) == 'fossil_electricity'.lower():
        txt = 'Fossil Electricity Generation in ' + str(h)
        color = 'Magma'
    if str(col) == 'renewables_electricity'.lower():
        txt = 'Renewable Electricity Generation in ' + str(h)
        color = 'Viridis'

    # Recursive function for map name headers:
    rename = humanize_string(col)

    # Plotly function to create map function:
    fig = pgo.Figure(data= pgo.Choropleth(
        locations = df['iso_code'],
        z = df[col],
        text = df['country'],
        autocolorscale = False,
        reversescale = True,
        marker_line_color = 'black',
        marker_line_width = 0.5,
        colorscale = color,
        colorbar_title = str(rename) + '; ' + 'TWh',
        zmax = maximum,
        zmin = 0
    ))
    fig.update_layout(
        title_text = str(txt),
        geo = dict(
            showframe = False,
            showcoastlines = False,
            scope = 'world',
            projection_type = 'equirectangular'
        ),
        annotations= [dict(
            x = 1,
            y = 1,
            xref = 'paper',
            yref = 'paper',
            text = 'From World Energy Consumption.csv',
            showarrow = False
        )]
    )

    fig.show()

    return fig

def plot_fossil_bar(s, e):
    """Uses matplotlib and seaborn to graph a bar chart. s = start date, e = end date."""
    # Data for these are all continuous from 1985 onward.

    # Filtering df from global statistics, filtering range into provided year set, and formatting text string to be used
    # in plot title.
    df = DF_fossil.loc[(DF['country'] == 'World')]
    df = df.loc[df['year'].isin(range(int(s), int(e)))]
    title1 = str('Global Fossil Electricity Generation between ' + str(s) + ' and ' + str(e))

    # Code to create and plot bar chart:
    sns.set(style= 'white')
    df.plot(x= 'year', kind= 'bar', stacked= True, color=['black', 'gold', 'slategray'])
    plt.title(title1, fontsize= 14)
    plt.xlabel('Years')
    plt.ylabel('TWh Generated per Year')
    figure = plt.show()

    return figure

def plot_renew_bar(s, e):
    """Uses matplotlib and seaborn to graph a bar chart. s = start date, e = end date."""
    # Data for these are all continuous from 1985 onward.

    # Filtering df from global statistics, filtering range into provided year set, and formatting text string to be used
    # in plot title.
    df = DF_renewable.loc[(DF['country'] == 'World')]
    df = df.loc[df['year'].isin(range(int(s), int(e)))]
    title1 = str('Global Renewable Electricity Generation between ' + str(s) + ' and ' + str(e))

    # Code to create and plot bar chart:
    sns.set(style= 'dark')
    df.plot(x= 'year', kind= 'bar', stacked= True, color=['green', 'dodgerblue', 'maroon', 'darkblue'])
    plt.title(title1, fontsize= 14)
    plt.xlabel('Years')
    plt.ylabel('TWh Generated per Year')
    figure = plt.show()

    return figure

def plot_mult_line_fossil(s, e):
    """Uses matplotlib to create a multiple line graph. s = start date, e = end date."""

    # Filtering df for usable column, and into years from provided year set:
    df = DF_regions[['country', 'year', 'fossil_electricity']]
    df = df.loc[(df['year'] >= int(s)) & (df['year'] <= int(e))]

    # Sets y-axis column as 'fossil_electricity,' followed by code to plot multiple line chart:
    df.set_index('year', inplace= True)
    df.groupby('country')['fossil_electricity'].plot(legend=True, xlabel= 'Year', ylabel= 'TWh Generation')

    plt.title('Fossil Electricity Generation by Region ' + str(s) + '-' + str(e))
    plt.grid()
    plt.xlim(2000, 2018)
    plt.xticks(ticks= range(int(s), int(e)), labels= range(int(s), int(e)), rotation= 90)
    fig = plt.show()

    return fig

def plot_mult_line_renew(s, e):
    """Uses matplotlib to create a multiple line graph. s = start date, e = end date"""

    # Filtering df for usable column, and into years from provided year set:
    df = DF_regions[['country', 'year', 'renewables_electricity']]
    df = df.loc[(df['year'] >= int(s)) & (df['year'] <= int(e))]

    # Sets y-axis column as 'fossil_electricity,' followed by code to plot multiple line chart:
    df.set_index('year', inplace= True)
    df.groupby('country')['renewables_electricity'].plot(legend=True, xlabel= 'Year', ylabel= 'TWh Generation')

    plt.title('Renewable Electricity Generation by Region ' + str(s) + '-' + str(e))
    plt.grid()
    plt.xlim(2000, 2018)
    plt.xticks(ticks= range(int(s), int(e)), labels= range(int(s), int(e)), rotation= 90)

    fig = plt.show()

    return fig

def growth_fossil(name):
    """uses statsmodels.api to make predictions and list them. USED IN PREDICTION FUNCTIONS."""
    # Recursive loop, used in function for predictions:
    # Name variable provided as region name from prediction function:
    df = DF_regions[DF_regions['country'] == name]
    df = df[df['year'].isin(range(1985, 2018))]

    X = df['year']
    Y = df['fossil_electricity']
    Xc = sm.add_constant(X)
    model = sm.OLS(Y, Xc).fit()

    return model.params[1], model.params[0]

def growth_renew(name):
    """uses statsmodels.api to make predictions and list them. USED IN PREDICTION FUNCTIONS."""
    # Recursive loop, used in function for predictions:
    # Name variable provided as region name from prediction function:
    df = DF_regions[DF_regions['country'] == name]
    df = df[df['year'].isin(range(1985, 2018))]

    X = df['year']
    Y = df['renewables_electricity']
    Xc = sm.add_constant(X)
    model = sm.OLS(Y, Xc).fit()

    return model.params[1], model.params[0]

def predict_fossil(year1, year2):
    """Plots a bar chart to predict fossil generation in 2030"""

    # Variables to set names of regions and names of color during the plotting function:
    names = list(set(DF_regions['country']))

    # For loop to cycle through regions, run the growth function, and calculate the slope/y-intercept of predictive line
    # this math is then used to plot the estimated bar chart for the year 2030:
    D1 = {}
    for name in names:
        m, b = growth_fossil(name)
        D1[name] = round(b + m * int(year1))

    D2 = {}
    for name in names:
        m, b = growth_fossil(name)
        D2[name] = round(b + m * int(year2))

    test = {
        "region": [],
        "current": [],
        "future": [],
    }

    for k, v in D1.items():
        test["region"].append(k)
        test["current"].append(v)

    for _, v in D2.items():
        test["future"].append(v)

    df = pd.DataFrame.from_dict(test, orient= 'columns')
    df = df.sort_values(by=['current', 'future'], ascending= False)

    sns.set(font_scale= 1)
    ay = df.plot(x= 'region', kind= 'bar', legend= False, color= ['sienna', 'orange'])
    ay.bar_label(ay.containers[0])
    ay.bar_label(ay.containers[1])
    plt.title('Predicted Fossil Electricity Generation in 2030')
    plt.xlabel('Region')
    plt.ylabel('Generation in TWh')
    plt.ylim(0, 12000)
    plt.xticks(rotation = 'horizontal', fontsize= 7)

    plt.show()

    return df

def predict_renew(year1, year2):
    """Plots a bar chart to predict fossil generation in 2030"""

    # Variables to set names of regions and names of color during the plotting function:
    names = list(set(DF_regions['country']))

    # For loop to cycle through regions, run the growth function, and calculate the slope/y-intercept of predictive line
    # this math is then used to plot the estimated bar chart for the year 2030:
    D1 = {}
    for name in names:
        m, b = growth_renew(name)
        D1[name] = round(b + m * int(year1))

    D2 = {}
    for name in names:
        m, b = growth_renew(name)
        D2[name] = round(b + m * int(year2))

    test = {
        "region": [],
        "current": [],
        "future": [],
    }

    for k, v in D1.items():
        test["region"].append(k)
        test["current"].append(v)

    for _, v in D2.items():
        test["future"].append(v)

    df = pd.DataFrame.from_dict(test, orient= 'columns')
    df = df.sort_values(by=['current', 'future'], ascending= False)

    sns.set(font_scale= 1)
    ay = df.plot(x= 'region', kind= 'bar', legend= False, color= ['green', 'palegreen'])
    ay.bar_label(ay.containers[0])
    ay.bar_label(ay.containers[1])
    plt.title('Predicted Renewable Electricity Generation in 2030')
    plt.xlabel('Region')
    plt.ylabel('Generation in TWh')
    plt.ylim(0, 2800)
    plt.xticks(rotation = 'horizontal', fontsize= 7)

    plt.show()

    return df

def res_check(region, fuel):
    """Regression plot to check validity. Accepts a region for statistical usage."""

    # Filters df by country, year, and provided fuel type, and checks that the data matches the region name:
    data = DF_regions[['year', 'country', str(fuel)]].copy()
    data = data[data.country.isin([str(region)])]
    data = data[data['year'].isin(range(1985, 2018))]

    # If check to set the title for the regression/residual chart:
    if fuel == 'fossil_electricity':
        title = 'Fossil'
    else:
        title = 'Renewable'

    # Plot regression/residual:
    sns.set(color_codes= True)
    plt.xlim(1985, 2020)
    sns.regplot(x = 'year', y = str(fuel), data = data, label = str(fuel))
    sns.residplot(x = 'year', y = str(fuel), data = data, label = 'residuals')
    plt.title(str(title) + ' Electricity Generation 1985-2020 in ' + str(region))
    plt.legend()

    figure = plt.show()

    return figure

def res_check_all_regions():
    """Recursive function to check residuals for all regions."""

    L = regions
    result = []

    for i in L:
        result.append(res_check(i, 'fossil_electricity'))
    for j in L:
        result.append(res_check(j, 'renewables_electricity'))

    return result

def describe_regions():
    """Returns the description for all regions, to acquire means, min's, max's, and quartiles."""

    df = DF_regions[['year', 'country', 'fossil_electricity', 'renewables_electricity']].copy()
    df = df[df['year'].isin(range(1985, 2018))]
    test = df.groupby('country').describe()

    slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(df['fossil_electricity'], df['year'])
    slope2, intercept2, r_value2, p_value2, std_err2 = scipy.stats.linregress(df['renewables_electricity'], df['year'])
    text1 = 'slope1 = ' + str(slope1), 'intercept1 = ' + str(intercept1) + ' r_value1 = ' + str(r_value1) + ' p_value1'\
            ' = ' + str(p_value1) + ' std_err1 = ' + str(std_err1)
    text2 = 'slope2 = ' + str(slope2), 'intercept2 = ' + str(intercept2) + ' r_value2 = ' + str(r_value2) + ' p_value2'\
            ' = ' + str(p_value2) + ' std_err2 = ' + str(std_err2)

    return test, text1, text2

# MAIN PROGRAM:
print(plot_map_one_year('fossil_electricity', 2000))
print(plot_map_one_year('fossil_electricity', 2019))
print(plot_map_one_year('renewables_electricity', 2000))
print(plot_map_one_year('renewables_electricity', 2019))
print(plot_fossil_bar(2000, 2019))
print(plot_renew_bar(2000, 2019))
print(plot_mult_line_fossil(2000, 2019))
print(plot_mult_line_renew(2000, 2019))
print(predict_fossil(2018, 2030))
print(predict_renew(2018, 2030))
print(res_check_all_regions())
print(describe_regions())
