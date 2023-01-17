import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objs as go
from  PIL import Image
import scipy as sp
#Add a logo (optional) in the sidebar
logo = Image.open(r'D:\Projects\strnet\Insights_Bees_logo.png')
st.sidebar.image(logo,  width=120)

#Add the expander to provide some information about the app
with st.sidebar.expander("About the App"):
     st.write("""
        This network graph app was built by Analysis Team. You can use the app to quickly generate an interactive network graph with different layout choices.
     """)

#Add a file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader('',type=['csv']) #Only accepts csv file format

#Add an app title. Use css to style the title
st.markdown(""" <style> .font {                                          
    font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Upload your data and generate an interactive network graph instantly...</p>', unsafe_allow_html=True)

tooltip_style = """
<style>
div[data-baseweb="tooltip"] {
  width: 150px;
}
</style>
"""
st.markdown(tooltip_style,unsafe_allow_html=True)




#Create the network graph using networkx
if uploaded_file is not None:     
    df=pd.read_csv(uploaded_file)  
    #df=pd.read_csv('C:\\Users\\PC\\Documents\\edgesc.csv')  

    A = list(df["Source"].unique())
    B = list(df["Target"].unique())
    node_list = set(A+B)
    #Gu = nx.Graph() #Use the Graph API to create an empty network graph object
    G = nx.DiGraph()
  
    #Add nodes and edges to the graph object
    for i in node_list:
        G.add_node(i)
    for i,j in df.iterrows():
        G.add_edges_from([(j["Source"],j["Target"])])    
 
    #Create three input widgets that allow users to specify their preferred layout and color schemes
    col1, col2, col3 = st.columns( [1, 1, 1])
    with col1:
        layout= st.selectbox('Choose a network layout',('Random Layout','Spring Layout','Shell Layout','Kamada Kawai Layout','Spectral Layout'))
    with col2:
        color=st.selectbox('Choose color of the nodes', ('Blue','Red','Green','Orange','Red-Blue','Yellow-Green-Blue'))      
    with col3:
        title=st.text_input('Add a chart title')

    #Get the position of each node depending on the user' choice of layout
    if layout=='Random Layout':
        pos = nx.random_layout(G) 
    elif layout=='Spring Layout':
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    elif  layout=='Shell Layout':
        pos = nx.shell_layout(G)            
    elif  layout=='Kamada Kawai Layout':
        pos = nx.kamada_kawai_layout(G) 
    elif  layout=='Spectral Layout':
        pos = nx.spectral_layout(G) 

    #Use different color schemes for the node colors depending on he user input
    if color=='Blue':
        colorscale='blues'    
    elif color=='Red':
        colorscale='reds'
    elif color=='Green':
        colorscale='greens'
    elif color=='Orange':
        colorscale='orange'
    elif color=='Red-Blue':
        colorscale='rdbu'
    elif color=='Yellow-Green-Blue':
        colorscale='YlGnBu'

    #Add positions of nodes to the graph
    for n, p in pos.items():
        G.nodes[n]['pos'] = p
    #Use plotly to visualize the network graph created using NetworkX
    #Adding edges to plotly scatter plot and specify mode='lines'
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1,color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    #Adding nodes to plotly scatter plot
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale=colorscale, #The color scheme of the nodes will be dependent on the user's input
            color=[],
            size=20,
            colorbar=dict(
                thickness=10,
                title='# Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=0)))

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color']=tuple([len(adjacencies[1])])+node_trace['marker']['color'] #Coloring each node based on the number of connections 
        node_info = str(adjacencies[0]) +' # of connections: '+str(len(adjacencies[1]))
        node_trace['text']+=tuple([node_info])

##############################################################################    
    #Plot the final figure
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=title, #title takes input from the user
                    title_x=0.45,
                    titlefont=dict(size=25),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    st.plotly_chart(fig, use_container_width=True) #Show the graph in streamlit




    #############################################################
    ####################################################
    # st.write(nx.average_shortest_path_length(G))
    # st.write(nx.diameter(G))
    # st.write(nx.density(G))
    # st.write(nx.average_clustering(G))
    # st.write(nx.transitivity(G))
    
    #########################################################
    # add_selectbox = st.sidebar.selectbox(
    # "How would you like to be contacted?",
    # ("Email", "Home phone", "Mobile phone"))

    # # Using "with" notation
    # with st.sidebar:
    #     add_radio = st.radio(
    #     "Choose a shipping method",
    #     ("Standard (5-15 days)", "Express (2-5 days)"))
        
    col1, col2 = st.columns(2)
    with col1:
        
       #</td></tr><tr><td>Average path length</td><td>"+str(round(nx.shortest_path_length(G),2))+"</td></tr>

        st.header("Network Basic Info")
        tooltip_style_table = "<table><tr><th>Attribute</th><th>Value</th><tr><td>Number of Users</td><td>" + str(len(G.nodes())) + "</td></tr><tr><td>Number of Edges</td><td>" + str(len(G.edges())) + "</td></tr><tr><td>Diameter</td><td>" +  str(max([max(j.values()) for (i,j) in nx.shortest_path_length(G)]) ) + "</table>"
        st.markdown(tooltip_style_table,unsafe_allow_html=True)
    with col2:
        st.header("Information Flow and clustering Coefficient")
        tooltip_style_table1 = "<table><tr><th>Attribute</th><th>Value</th><tr><td>Network Density</td><td>" + str(round(nx.density(G),2)) + "</td></tr><tr><td>Clustering Coefficient</td><td>" + str(round(nx.average_clustering(G),2)) + "</td></tr><tr><td>Network Transitivity</td><td>" + str(round(nx.transitivity(G),2)) + "</td></tr>"
        st.markdown(tooltip_style_table1,unsafe_allow_html=True)
#         st.write(pd.DataFrame({
#     'Attribute': ['Network Density', 'Number of Edges', 'Clustering Coefficient', 'Network Transitivity'],
#     'Value': [nx.density(G), nx.average_clustering(G), nx.average_clustering(G), nx.transitivity(G)],
# }))
        
     # sorted(bet_cen.items(), key=lambda x:x[1], reverse=True)[0:5]
     # sorted(deg_cen.items(), key=lambda x:x[1], reverse=True)[0:5]
     # sorted(page_rank.items(), key=lambda x:x[1], reverse=True)[0:5]
     # sorted(clos_cen.items(), key=lambda x:x[1], reverse=True)[0:5]   

    # ####################################################
    # Compute the closeness centrality of G: clos_cen
    # st.write('Top 5 page rank users')
    # clos_cen = nx.closeness_centrality(G)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        deg_cen = nx.degree_centrality(G)
        deg_cen_new = pd.DataFrame({'User': list(deg_cen.keys()), 'Value': list(deg_cen.values())})
        st.header("Top 5 Central Users")
        original_title = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Original image</p>'
        st.markdown(original_title, unsafe_allow_html=True)
        #st.write('''_Hello''')
        #st.write(deg_cen_new.sort_values(by='Value',ascending=False))   
        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        # Display a static table
        st.table(deg_cen_new.sort_values(by='Value',ascending=False).head(10))

    with col2:
        # Compute the betweenness centrality of G: bet_cen
        bet_cen = nx.betweenness_centrality(G)
        bet_cen_new = pd.DataFrame({'User': list(bet_cen.keys()), 'Value': list(bet_cen.values())})
        st.header("Top 5 Between users")
        #st.write(bet_cen_new.sort_values(by='Value',ascending=False))
        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        # Display a static table
        st.table(bet_cen_new.sort_values(by='Value',ascending=False).head(10))

    with col3:
        # Compute the page rank of G: page_rank
        page_rank = nx.pagerank(G)
        page_rank_new = pd.DataFrame({'User': list(page_rank.keys()), 'Value': list(page_rank.values())})
        st.header("Top 5 page rank users")
        #st.write(page_rank_new.sort_values(by='Value',ascending=False))
        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        # Display a static table
        st.table(page_rank_new.sort_values(by='Value',ascending=False).head(10))
        

    ######################Centrality Measures###########################

    # # Compute the betweenness centrality of G: bet_cen
    # st.write(bet_cen = nx.betweenness_centrality(G))
    # # Compute the degree centrality of G: deg_cen
    # st.write(deg_cen = nx.degree_centrality(G))
    # # Compute the page rank of G: page_rank
    # st.write(page_rank = nx.pagerank(G))
    # # Compute the closeness centrality of G: clos_cen
    # st.write(clos_cen = nx.closeness_centrality(G))

    

    ###############################################

    #################################################
    # get all cliques
    all = nx.find_cliques(G)
    st.write(all)
    # get the largest clique
    largest_clique = sorted(nx.find_cliques(G), key=lambda x:len(x))[-1]
    st.write(largest_clique)
   
