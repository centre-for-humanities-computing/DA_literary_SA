
''''''
# functions for sentiment analysis (dictionary-based methods)
# and for plotting (adjusted from figs.py)
# for the Danish HCA SA study

''''''

import os
from utils import *



# normalization
def normalize(ts, scale_zero_to_ten=False):
    ts_min = np.min(ts)
    ts_max = np.max(ts)
    ts01 = (ts - ts_min) / (ts_max - ts_min)  # Normalizes to 0-1 scale
    ts10 = ts01 * 10  # Converts to 0-10 scale
    
    if scale_zero_to_ten:
        return ts10
    else:
        return ts01

    
# Function to safely convert to float
def convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


## plotting DISTRIBUTIONS
def plot_kdeplots_or_histograms(df, scores_list, type, plottitle, plts_per_row, l, h):
    plots_per_row = plts_per_row

    if len(scores_list) <= plots_per_row:
        fig, axes_list = plt.subplots(1, len(scores_list), figsize=(l, h), dpi=300)#, sharey=True)
    else:
        rows = len(scores_list) // plots_per_row
        if len(scores_list) % plots_per_row != 0:
            rows += 1
        fig, axes_list = plt.subplots(rows, plots_per_row, figsize=(l, h * rows), dpi=300)#, sharey=True)
        
    fig.tight_layout(pad=3)

    labels = [x.replace('_', ' ') for x in scores_list]

    for i, score in enumerate(scores_list):
        sns.set_style(style="whitegrid", font_scale=2, font='serif')

        ax = axes_list.flat[i]

        if type == 'histplot':
            if labels[i].startswith('tr'):
                sns.histplot(data=df[score], ax=ax, color='#38a3a5')
            elif labels[i].lower() == 'human':
                sns.histplot(data=df[score], ax=ax, color='lightgrey')
            else: 
                sns.histplot(data=df[score], ax=ax, color='lightcoral')
        else:
            sns.kdeplot(data=df[score], ax=ax, log_scale=False, color='#38a3a5')

        # Set labels
        ax.set_xlabel(labels[i])
        
        # if i >= 1:
        #     ax.set_ylabel('')  # Set the y-axis label to an empty string
        
    fig.suptitle(plottitle, fontsize=20)
    plt.tight_layout()
    
    if os.path.exists('figures') == True:
        save_title = plottitle.split(' ')[:3]
        save_title = '_'.join(save_title)
        plt.savefig(f'figures/{save_title}_distribution.png')
        
    plt.show()
    return fig



# plotting BOXPLOTS for comparing two gorups
def pairwise_boxplots_canon(df, measures, category, category_labels, plottitle, outlier_percentile, h, w, remove_outliers=False, save=False):
# Only works for 5 boxplots for now!

    plots_per_row = len(measures) # just for now make number that are passed

    if len(measures) <= plots_per_row:
        fig, axes = plt.subplots(1, len(measures), figsize=(w, h), dpi=300)
    else:
        num_rows = math.ceil(len(measures) / plots_per_row)

        fig, axes = plt.subplots(num_rows, len(measures), figsize=(18, 8), dpi=300) # (18, 8 * rows), dpi=300)

    cat1_df = df.loc[df[category] == 1]
    cat2_df = df.loc[df[category] != 1]

    labels = [x.split('_')[1].lower() for x in measures]

    # Iterate over the significant columns
    for i, column in enumerate(measures):
        ax = axes[i]
        #df_dfered = df.loc[df[column].notnull()]
        cat1_df = cat1_df.loc[cat1_df[column].notnull()]
        cat2_df = cat2_df.loc[cat2_df[column].notnull()]
        
        # Boxplot
        ax.boxplot([cat1_df[column], cat2_df[column]],
                labels=category_labels,
                boxprops=dict(alpha=1, linewidth=1),
                widths=[0.75, 0.75], showfliers=False)
        ax.set_ylabel(labels[i], fontsize=24)


        # Scatterplot within boxplot
        colors = ['#C1666B', '#38a3a5']

        for j, group in enumerate([cat1_df, cat2_df]):
            column_data = group[column]

            if remove_outliers == True:
                # Calculate the 99.5th percentile
                percentile_95 = np.percentile(column_data, outlier_percentile)
                # dfer data points
                data = group[column][group[column] <= percentile_95]
            else:
                data = group[column]
            
            # creating random x coordinates to plot as a bulk
            x = np.random.normal(j + 1, 0.12, size=len(data))
            # Plot scatterpoints
            ax.plot(x, data, '.', alpha=0.65, color=colors[j], markersize=10)

    fig.suptitle(f'{plottitle}', fontsize=24)
    sns.set_style("whitegrid")
    plt.tight_layout()
    if save == True:
        plt.savefig(f'figures/features_boxplot_{plottitle}.png')
    # Show the plot
    plt.show()
    return fig

# plotting CEDs
## function to calculate KS test for two samples
def get_kstest(implicit_df, explicit_df, measure_list, labels):
    stats_all = []

    for i, measure in enumerate(measure_list):
        values_impl = [x for x in implicit_df[measure] if not pd.isna(x)]
        values_expl = [x for x in explicit_df[measure] if not pd.isna(x)]

        #a, b = [e[0] for e in measure[0]], [e[0] for e in measure[1]]
        ks_stat, ks_p_value = stats.ks_2samp(values_impl, values_expl)
        stats_all.append([round(ks_stat,3), round(ks_p_value,3)])
        print(f'{labels[i]} - KS Statistic: {ks_stat}, p-value: {ks_p_value}')

    return stats_all

## compute cdf
def compute_cdf(data):
        n = len(data)
        x = np.sort(data)
        y = np.arange(1, n+1) / n
        return x, y


### CED plot
def ced_plot(implicit_df, explicit_df, measure_list, labels, save=False, save_title=False):

    stats_all = get_kstest(implicit_df, explicit_df, measure_list, labels)

    apos = '**' # for p-value < 0.01

    fig, axes = plt.subplots(1, len(measure_list), figsize=(22, 4), sharey=True, dpi=500)

    for i, measure in enumerate(measure_list):
        #a, b = [e[0] for e in measure[0]], [e[0] for e in measure[1]]
        values_impl = implicit_df[measure]
        values_expl = explicit_df[measure]
        print(len(values_impl), len(values_expl))

        # Calculate CDF for each population
        x_a, y_a = compute_cdf(values_impl)
        x_b, y_b = compute_cdf(values_expl)

        sns.set_theme(style="whitegrid", font_scale=1.5)
        # Plotting CDF
        axes[i].plot(x_a, y_a, marker='.', markersize=8, alpha=0.45, linestyle='none', label='' if i > 0 else 'Implicit')
        axes[i].plot(x_b, y_b, marker='.', markersize=8, alpha=0.45, linestyle='none', label='' if i > 0 else 'Explicit')

        #axes[i].set_title(f'CED {labels[i]}')
        axes[i].set_title(f'CED {labels[i]}, KS: {stats_all[i][0]}{apos if stats_all[i][1] < 0.01 else ""}')

        axes[i].set_xlabel(f'{labels[i]}')

        # if i < 3:
        #     axes[i].legend_.remove()

    axes[0].set_ylabel('Cumulative Probability')

    axes[0].legend()  # Adding legend to the last subplot

    plt.tight_layout()

    if save == True:
            # if save title exists
        if save_title:
            plt.savefig(f'figures/{save_title}_{str(len(measure_list))}_measures_CED.png')
        else:
            plt.savefig(f'figures/{str(len(measure_list))}_measures_CED.png')
    plt.show()


# Histplot, two groups
def histplot_two_groups(implicit_df, explicit_df, measure_list, labels, l, h, title_plot, density=False, save=False, save_title=False):

    sns.set_theme(style="whitegrid", font_scale=1.5)
    fig, axes_list = plt.subplots(1, len(measure_list), figsize=(l, h), dpi=300)#, sharey=True)
    
    for i, measure in enumerate(measure_list):
        ax = axes_list.flat[i]
        if density == True:
            sns.histplot(data=implicit_df, x=measure, ax=ax, color='blue', kde=True, label='implicit', stat='density')
            sns.histplot(data=explicit_df, x=measure, ax=ax, color='red', kde=True, label='explicit', stat='density')
        else:
            sns.histplot(data=implicit_df, x=measure, ax=ax, color='blue', kde=True, label='implicit')
            sns.histplot(data=explicit_df, x=measure, ax=ax, color='red', kde=True, label='explicit')

        ax.set_xlabel(f'{labels[i]}')

        # if i < 3:
        #     axes[i].legend_.remove()

    axes_list[0].legend()  # Adding legend to the last subplot

    plt.tight_layout()
    
    # set title
    if title_plot != None:
        fig.suptitle(title_plot, fontsize=20)

    if save == True:
            # if save title exists
        if save_title:
            plt.savefig(f'figures/{save_title}_{str(len(measure_list))}_distributions.png')
        else:
            plt.savefig(f'figures/{str(len(measure_list))}_distributions.png')
    plt.show()




# Plotting

###
#
# Plotly visualisation of a correlation,
# takes a first measure, a second measure and colors canonical works if canon == True
def plotly_viz_correlation_improved(df, first, second, canon_col_name, w, h, hoverdata_column, color_canon=False, save=False):

    # make the labels
    labels = {first:str(first).replace('_', ' ').lower(), second:str(second).replace('_', ' ').lower(),
            'TITLE':'title','AUTH_LAST':'author'}
    
    # remove the very outliers if plotting sentence length
    if first == 'AVG_SENTLEN':
        dat = df.loc[df['average_sentlen'] < 500]
    else:
        dat = df

    if second == 'READABILITY_FLESCH_EASE':
        dat = df.loc[df['READABILITY_FLESCH_EASE'] > 0]
    else:
        dat = df

    ## Correlation
    # remove 0 values to do the correlation
    #df = dat[(dat[first] != 0) & (dat[second] != 0)]
    df = dat[(dat[first].notna()) & (dat[second].notna())]
    print('number of sentences considered: ', len(df))

    # Get spearman r and make the coeff the title of the plot
    coef, pvalue = stats.spearmanr(df[first], df[second])

    if pvalue < 0.01:
        pvalue_viz = 'p < 0.01'
    elif pvalue < 0.05:
        pvalue_viz = 'p < 0.05'
    else:
        pvalue_viz = 'p > 0.05!'
    # Set this as title
    title = "Spearman's r (" + str(round(coef, 3)) + ", " + pvalue_viz + ')'

    # We also want the corr of the canon if color_canon == True
    if color_canon == True:
        canon_only_df = dat.loc[dat['CANON_ALL'] == 1]
        # remove 0 values to do the correlation
        df_canon = canon_only_df[(canon_only_df[first].notnull()) & (canon_only_df[second].notnull())]
        print('number of titles considered: ', len(df_canon))

        # Get spearman r and make the coeff the title of the plot
        coef_canon, pvalue_canon = stats.spearmanr(df_canon[first], df_canon[second])

        if pvalue_canon < 0.01:
            pvalue_viz_canon = 'p < 0.01'
        elif pvalue_canon < 0.05:
            pvalue_viz_canon = 'p < 0.05'
        else:
            pvalue_viz_canon = 'p > 0.05!'
        # Set this as title

        subtitle = "for canon only (" + str(round(coef_canon, 3)) + ", " + pvalue_viz_canon + ')'

    # Define colors
    colorsId = {'1': '#e377c2', '0': '#1f77b4'}


    ## Plot
    if color_canon == True:
        fig = px.scatter(dat, x=first, y=second, hover_data= {'CANON_ALL':False, 'TITLE':True, 'AUTH_LAST':True}, #['TITLE_MODERN', 'AUTH_LAST_MODERN'], 
                        opacity=0.6, #marginal_x="histogram", #marginal_y="histogram", 
                        title=f"{title}<br><sup>{subtitle}</sup>", labels=labels, 
                        #color_discrete_sequence=px.colors.qualitative.Dark24, 
                        color='CANON_ALL', symbol="CANON_ALL", 
                        width=w, height=h, color_discrete_sequence=list(colorsId.values()))
        
    if color_canon == False:
        fig = px.scatter(dat, x=first, y=second, hover_data= {hoverdata_column:True}, #hover_data=['TITLE_MODERN', 'AUTH_LAST_MODERN'], 
                    opacity=0.4, #marginal_x="histogram", #marginal_y="histogram", 
                    title=title, labels=labels, 
                    width=w, height=h, color_discrete_sequence=list(colorsId.values()))#,color_discrete_sequence=px.colors.qualitative.Dark24)

    # layout
    fig.update_layout(
        font_family="Courier New",
        font_color="black",
        title=dict(font=dict(size=15), yref='paper', x=0.3),
        margin=dict(l=70, r=50, t=50, b=60),
        #yaxis_range=[0,1100], xaxis_range=[0,5]
    )

    #fig.update_traces(marker={'size': 8}, line=dict(color="black", width=0.5)) #, 'color':list(colorsId.values())

    fig.update_coloraxes(showscale=False)

    fig.show()

    if save == True:
        if os.path.exists('figures') == True:
            fig.write_html(f'figures/{first}_{second}_scatterplot.html')
        else:
            print('Sucker. Please create a folder called figures in the directory where you want to save the plots')

    return fig



# Adding plotting scatteplots function
def plot_scatters(df, scores_list, var, color, w, h, hue=False, remove_outliers=False, outlier_percentile=100, show_corr_values=False):
    num_plots = len(scores_list)
    num_rows = 1
    num_cols = num_plots // num_rows

    labels = [x.replace('_', ' ').lower() for x in scores_list]

    fig, axes_list = plt.subplots(num_rows, num_cols, figsize=(w, h))
    axes_list = axes_list

    if hue == None:
        hue = 'False'

    for index, score in enumerate(scores_list):
        df = df.loc[df[score].notnull()]

        if remove_outliers == True:
            percentile = np.percentile(df[score], outlier_percentile)
            df = df.loc[df[score] <= percentile]
        
        sns.scatterplot(data=df, x=var, y=score, ax=axes_list[index],
                        color=color, hue = hue, s= 35, alpha= 0.3, palette='rocket')
            
        # I want to add the spearman corr as title of each sublot
        if show_corr_values == True:
            
            check = df.loc[df[var].notnull()]

            correlation = stats.spearmanr(check[var], check[score])
            corr_value = round(correlation[0], 3)

            if correlation[1] < 0.01:
                axes_list[index].set_title(f"Spearm. coef: {corr_value}, p<0.01", fontsize=15)
            if correlation[1] >= 0.01:
                axes_list[index].set_title(f"Spearm. coef: {corr_value}, OBS: p>0.01", fontsize=15)

            print(f'corr_spearman of {score} & {var}:', round(correlation[0], 3), 'pval:', correlation[1])

        axes_list[index].set_ylabel(labels[index], fontsize=20)
        axes_list[index].set_xlabel(var.replace('_', ' ').lower(), fontsize=20)
        #axes_list[index].set_ylim(bottom=0)

        fig.tight_layout(pad=1)

    print("mæhmæhmnæh")

    plt.show()





# For SENTIMENT ANALYSIS

# to convert transformer scores to the same scale as the dictionary-based scores
def conv_scores(lab, sco, spec_lab): #insert exact labelnames in order positive, negative og as positive, neutral, negative
    
    converted_scores = []
    
    if len(spec_lab) == 2:
        spec_lab[0] = "positive"
        spec_lab[1] = "negative"

        for i in range(0, len(lab)):
            if lab[i] == "positive":
                converted_scores.append(sco[i])
            else:
                converted_scores.append(-sco[i])
            
    if len(spec_lab) == 3:
        spec_lab[0] = "positive"
        spec_lab[1] = "neutral"
        spec_lab[2] = "negative"
        
        for i in range(0, len(lab)):
            if lab[i] == "positive":
                converted_scores.append(sco[i])
            elif lab[i] == "neutral":
                converted_scores.append(0)
            else:
                converted_scores.append(-sco[i])
    
    return converted_scores


# and VADER 
sid =  SentimentIntensityAnalyzer()

def sentimarc_vader(text, untokd=True):
    if untokd:
        sents = nltk.sent_tokenize(text)
        print(len(sents))
    else: sents = text
    arc=[]
    for sentence in sents:
        compound_pol = sid.polarity_scores(sentence)['compound']
        arc.append(compound_pol)
    return arc
