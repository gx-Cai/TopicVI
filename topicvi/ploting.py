import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def ridge_topic_group(
    fdata, 
    groupby,
    target_topic, 
    topic_annote,
    target_clusters = None,
    paltte = None,
    figsize=None,
    hspace = 0.1,
    label_relative_x = 0.8,
    ylim = None,
    prefix = 'Cluster',
    **kde_kwargs
):
    target_clusters = target_clusters or fdata.obs[groupby].unique()
    paltte = paltte or 'tab20'

    cluster_vect = fdata.obs[groupby].isin(target_clusters)
    cluster_obs = fdata.obs[cluster_vect]

    def make_topic_loading(i,):
        topic_loading = fdata.obs_vector(f'topic_{i}')[cluster_vect]
        topic_loading = pd.DataFrame(
                {
                    'loading': topic_loading,
                    'cluster': cluster_obs[groupby].astype(str),
                }, 
                index=cluster_obs.index
            )
        return topic_loading

    topic_loading = make_topic_loading(target_topic)

    g = sns.FacetGrid(
        topic_loading, row='cluster', hue='cluster', 
        aspect=3, height=1.5, 
        palette=paltte,
        sharex=True, sharey=True,
        row_order=target_clusters,
    )
    g.map(
        sns.kdeplot, 'loading', 
        clip_on=False,
        alpha=0.8, lw=1.5, fill=True, **kde_kwargs
    )
    # g.map(plt.axhline, y=0, lw=1, color='black')
    g.refline(y=0, linewidth=2, linestyle='-', color=None, clip_on=False)
    g.set_titles('')
    g.set(yticks=[], ylabel='')
    g.despine(left=True, bottom=True)

    def set_label(x, color, label):
        ax = plt.gca()
        maxx = g.axes[-1, -1].get_xlim()[1]
        ax.text(maxx*label_relative_x, 0.3,prefix + label, fontweight='semibold', color=color, 
                ha='left', va='bottom', fontsize=12)
    g.map(set_label, 'loading')
    if figsize:
        g.fig.set_size_inches(figsize[0], figsize[1])
    else:
        g.fig.set_figheight(len(target_clusters) * .35)
        g.fig.set_figwidth(len(target_clusters) * 0.75)

    g.fig.subplots_adjust(hspace=hspace)
    g.set_xlabels('Topic Loading', )
    # with a title
    g.fig.suptitle(
        f'Topic {target_topic}\n({topic_annote})', fontsize=12,
        verticalalignment='bottom', horizontalalignment='center',
    )

    if ylim:
        g.set(ylim=ylim)
    return g


def nested_pie_chart(
    topic_list, prior_annote, layer_annote,
    fig_size=(6, 6), 
    hole_size=0.3, 
    layer_gap=0.05,
    cmap = 'Set2',
    annote_genes = None,
):
    """
    Create a nested pie chart with multiple layers and gaps between layers.
    
    Parameters:
    - topic_list: List of genes to be represented in the pie chart
    - prior_annote: Dictionary mapping prior annotations to genes
    - layer_annote: List of annotations for each layer, list of strings or 
    - cmap: Color map for the pie chart
    - fig_size: Size of the figure
    - hole_size: Size of the hole in the center (0 to 1)
    - layer_gap: Gap between concentric layers (0 to 1)
    
    Returns:
    - fig, ax: The figure and axis objects
    """

    # Prepare data and colors for the nested pie chart

    n_genes = len(topic_list)
    num_layers = len(layer_annote)
    all_data = [np.ones(n_genes) for _ in range(num_layers)]

    all_related_prior = []
    for pi in layer_annote:
        if isinstance(pi, str):
            all_related_prior.append(pi)
        elif isinstance(pi, list):
            all_related_prior.extend(pi)
        else:
            raise ValueError(f"prior {pi} not in proper format")
        
    all_related_prior = list(set(all_related_prior))+['Other']
    palette = list(sns.color_palette(cmap, n_colors=len(all_related_prior)-1)) + ['.9'] # other in grey color
    palette = {
        k: palette[i] for i, k in enumerate(all_related_prior)
    }
    
    all_colors = []
    for li, layer in enumerate(layer_annote):
        others_color = palette['Other'] if li==num_layers-1 else 'white' 
        if isinstance(layer, str):
            all_colors.append(
                [palette[layer] if k in prior_annote[layer] else others_color for k in topic_list],
            )
        elif isinstance(layer, list):
            layer_color = []
            for k in topic_list:
                flag = False
                for t in layer:
                    if k in prior_annote[t]:
                        layer_color.append(palette[t])
                        flag = True
                        break
                if not flag:
                    layer_color.append(others_color)
            all_colors.append(layer_color)
    fig, ax = plt.subplots(figsize=fig_size)
        
    # Calculate the width of each layer, accounting for gaps
    total_width_available = 1.0 - hole_size
    total_gaps_width = layer_gap * (num_layers - 1)
    effective_width = total_width_available - total_gaps_width
    layer_width = effective_width / num_layers
    
    # Start with the outer radius
    outer_radius = 1.0
    
    # Create the pie chart for each layer
    for i, (layer_data, layer_colors) in enumerate(zip(all_data, all_colors)):
        inner_radius = outer_radius - layer_width
        
        # Create the wedges for this layer
        wedges, _ = ax.pie(
            layer_data,
            radius=outer_radius,
            colors=layer_colors,
            wedgeprops=dict(width=layer_width, edgecolor='w', linewidth=0.5),
            startangle=90
        )
        
        # Update the outer radius for the next layer, adding the gap
        outer_radius = inner_radius - layer_gap
    
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in all_related_prior],
        labels=all_related_prior,
        bbox_to_anchor=(1, 0.75),
        frameon=False,
        ncols=1, fontsize=10,
    )

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_aspect('equal')

    # annote the genes in wedges
    if annote_genes:
        for gene in annote_genes:
            gene_idx = topic_list.index(gene)
            wedge = wedges[gene_idx]
            angle = (wedge.theta1 + wedge.theta2) / 2
            x = np.cos(np.radians(angle)) * (outer_radius - layer_width / 2 + 1.5*layer_gap)
            y = np.sin(np.radians(angle)) * (outer_radius - layer_width / 2 + 1.5*layer_gap)
            ax.text(x, y, gene, ha='center', va='center', fontsize=6, rotation=angle)

    return fig, ax, wedges
