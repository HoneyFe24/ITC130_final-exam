import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('Raisin_Dataset.csv')

df = load_data()

# Title of the web app
st.title('Raisin Dataset Analysis')

# Adding some text to the app
st.write("""
This app showcases various insights from the Raisin Dataset.
The dataset includes measurements of raisin properties and their classification.
""")

# Sidebar with additional text
st.sidebar.write("### ITC130 FINAL EXAM")
st.sidebar.write("**Presented By:**")
st.sidebar.write("Nicole Anne Cabantac and Honey Fe Sacayan")

# Sidebar for navigation
st.sidebar.title("Navigation")
visualization = st.sidebar.selectbox("Choose a visualization", ["Dataset Overview", "Pairplot", "Class Distribution", "Feature Means", "Scatter Plot", "Correlation Matrix", "Box Plot", "Violin Plot", "Histogram", "Swarm Plot", "PairGrid"])

# Sidebar for selecting columns to visualize
st.sidebar.write("## Select Columns for Visualization")
columns = df.columns[:-1]  # Exclude the target column 'Class'
selected_columns = st.sidebar.multiselect("Select columns to plot", columns)

# Slider for selecting range of rows to display
st.sidebar.write("## Select Range of Rows")
row_start = st.sidebar.slider("Start row", 0, len(df)-1, 0)
row_end = st.sidebar.slider("End row", 1, len(df), len(df))

# Filtering the dataframe based on slider values
filtered_df = df.iloc[row_start:row_end]

# Display the dataframe
if visualization == "Dataset Overview":
    st.write("## Raisin Dataset")
    st.dataframe(filtered_df)

# Pairplot for selected columns
if visualization == "Pairplot":
    if selected_columns:
        st.write("## Pairplot of Selected Features")
        st.write("""
        The pairplot shows pairwise relationships in the dataset. Each scatter plot displays the relationship between two features, colored by the class of the raisin.
        """)
        pairplot_fig = sns.pairplot(filtered_df, vars=selected_columns, hue='Class', markers=["o", "s"])
        st.pyplot(pairplot_fig)
    else:
        st.warning("Please select at least one column for the pairplot.")

# Pie chart of class distribution
if visualization == "Class Distribution":
    st.write("## Class Distribution")
    st.write("""
    The pie chart displays the distribution of the two raisin classes in the dataset.
    """)
    class_distribution = df['Class'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)

# Bar chart of feature means grouped by class
if visualization == "Feature Means":
    if selected_columns:
        st.write("## Mean Values of Selected Features by Class")
        st.write("""
        The bar chart shows the mean values of the selected features, grouped by raisin class. This helps to compare the average measurements of the features between the two classes.
        """)
        mean_values = df.groupby('Class')[selected_columns].mean().reset_index()
        st.bar_chart(mean_values.set_index('Class'))
    else:
        st.warning("Please select at least one column for the bar chart.")

# Scatter plot of two selected features
if visualization == "Scatter Plot":
    st.write("## Scatter Plot of Two Features")
    st.write("""
    The scatter plot visualizes the relationship between two selected features, with points colored by the raisin class. This helps to identify patterns or differences between the classes.
    """)
    x_feature = st.selectbox("Select X axis feature", columns, key='x_feature')
    y_feature = st.selectbox("Select Y axis feature", columns, key='y_feature')
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=filtered_df, x=x_feature, y=y_feature, hue='Class', style='Class', markers=["o", "s"], ax=ax2)
    st.pyplot(fig2)

# Display correlations excluding the 'Class' column
if visualization == "Correlation Matrix":
    st.write("## Correlation Matrix")
    st.write("""
    The correlation matrix shows the Pearson correlation coefficients between features. It helps to identify the strength and direction of the linear relationship between pairs of features.
    """)
    corr = df.drop(columns=['Class']).corr()
    fig3, ax3 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

# Box plot for selected columns
if visualization == "Box Plot":
    if selected_columns:
        st.write("## Box Plot of Selected Features by Class")
        st.write("""
        The box plot displays the distribution of the selected features for each raisin class. It shows the median, quartiles, and potential outliers, helping to understand the spread and central tendency of the data.
        """)
        for col in selected_columns:
            fig4, ax4 = plt.subplots()
            sns.boxplot(x='Class', y=col, data=filtered_df, ax=ax4)
            ax4.set_title(f'Box Plot of {col} by Class')
            st.pyplot(fig4)
    else:
        st.warning("Please select at least one column for the box plot.")

# Violin plot for selected columns
if visualization == "Violin Plot":
    if selected_columns:
        st.write("## Violin Plot of Selected Features by Class")
        st.write("""
        The violin plot combines aspects of the box plot and density plot, showing the distribution and probability density of the selected features for each raisin class.
        """)
        for col in selected_columns:
            fig5, ax5 = plt.subplots()
            sns.violinplot(x='Class', y=col, data=filtered_df, ax=ax5)
            ax5.set_title(f'Violin Plot of {col} by Class')
            st.pyplot(fig5)
    else:
        st.warning("Please select at least one column for the violin plot.")

# Histogram for selected columns
if visualization == "Histogram":
    if selected_columns:
        st.write("## Histogram of Selected Features")
        st.write("""
        The histogram displays the distribution of the selected features, with different colors representing the raisin classes. It helps to see the frequency and distribution of feature values within each class.
        """)
        for col in selected_columns:
            fig6, ax6 = plt.subplots()
            sns.histplot(filtered_df, x=col, hue='Class', multiple='stack', ax=ax6)
            ax6.set_title(f'Histogram of {col}')
            st.pyplot(fig6)
    else:
        st.warning("Please select at least one column for the histogram.")

# Swarm plot for selected columns
if visualization == "Swarm Plot":
    if selected_columns:
        st.write("## Swarm Plot of Selected Features by Class")
        st.write("""
        The swarm plot shows the distribution of individual data points for the selected features, grouped by raisin class. It provides a clear visualization of data density and potential clusters.
        """)
        for col in selected_columns:
            fig7, ax7 = plt.subplots()
            sns.swarmplot(x='Class', y=col, data=filtered_df, ax=ax7)
            ax7.set_title(f'Swarm Plot of {col} by Class')
            st.pyplot(fig7)
    else:
        st.warning("Please select at least one column for the swarm plot.")

# PairGrid for selected columns
if visualization == "PairGrid":
    if selected_columns:
        st.write("## PairGrid of Selected Features")
        st.write("""
        The PairGrid provides a grid of plots to visualize pairwise relationships between the selected features. The diagonal plots show the distribution of each feature, while the off-diagonal plots show the relationships between pairs of features.
        """)
        pairgrid_fig = sns.PairGrid(filtered_df, vars=selected_columns, hue='Class')
        pairgrid_fig = pairgrid_fig.map_diag(sns.histplot)
        pairgrid_fig = pairgrid_fig.map_offdiag(sns.scatterplot)
        pairgrid_fig = pairgrid_fig.add_legend()
        st.pyplot(pairgrid_fig)
    else:
        st.warning("Please select at least one column for the PairGrid.")

# Narrative
st.write("""
## Data Context and Usage
The Raisin Dataset contains various measurements of raisin properties. 
The target variable, 'Class', indicates the type of raisin. The features include:
- **Area**: Area of the raisin
- **MajorAxisLength**: Length of the major axis of the raisin
- **MinorAxisLength**: Length of the minor axis of the raisin
- **Eccentricity**: Eccentricity of the raisin
- **ConvexArea**: Convex area of the raisin
- **Extent**: Extent of the raisin
- **Perimeter**: Perimeter of the raisin

The visualizations provide insights into the distribution and relationships between these features, helping to understand how they contribute to classifying the raisin type.
""")
