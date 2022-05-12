#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lefteris

@subject: sc app
"""

# Import required libraries
import pandas as pd
import numpy as np
import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
from supply_demand_opt import SupplyDemand
import plotly.graph_objects as go
from math import sin, cos, sqrt, atan2, radians



#A function to calculate distance
def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

df = pd.read_csv("data/demand.csv", sep='\s+')
D = np.array(df.iloc[-1,1:-2])
S = np.array(df.iloc[:-1,-2])
c = np.array(df.iloc[:-1,-1])

#Create random coordinates
coord_init =   [39.553464, 21.759884] #lat, lon
N, M = S.shape[0], D.shape[0]
np.random.seed(0)
plant_loc =  pd.DataFrame({
    'pl_id' : range(N),
    'lon_pl' : (-1 + 2*np.random.random(N))*10 +coord_init[1],
    'lat_pl' : (-1 + 2*np.random.random(N))*10 +coord_init[0]
})
warehouse_loc = pd.DataFrame({
    'wr_id' : range(M),
    'lon_wr' : (-1 + 2*np.random.random(M))*10 +coord_init[1],
    'lat_wr' : (-1 + 2*np.random.random(M))*10 +coord_init[0]
})

C = np.zeros((N,M))
for i in range(N):
    for j in range(M):
        C[i,j] = calculate_distance(plant_loc.iloc[i,2],
                                   plant_loc.iloc[i,1],
                                   warehouse_loc.iloc[j,2],
                                   warehouse_loc.iloc[j,1])



supp = SupplyDemand(C,D,S,c)
report = supp.get_report()
min_supply=report['supply'].min()
max_supply=report['supply'].max()
pl_id_list = list(set(report['pl_id'].sort_values()))
wr_id_list = list(set(report['wr_id'].sort_values()))





external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


def create_card(card_id, title,value = '100', description=''):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4(title, id=f"{card_id}-title"),
                html.H2(value, id=f"{card_id}-value"),
                html.P(description, id=f"{card_id}-description")
            ]
        )
    )


app.layout = html.Div(
    [
        dbc.Row([
            dbc.Col([create_card("C_cost_card", "Transportation Cost",
                                 value=dcc.Markdown(id="C_cost_card"))]),
            dbc.Col([create_card("c_cost_card", "Operations Cost",
                                 value=dcc.Markdown(id="c_cost_card"))]),
            dbc.Col([create_card("volume_card", "Total Volume",
                                 value=dcc.Markdown(id="volume_card"))])
            ]),
        dbc.Row([
            dbc.Col([create_card("filter_plant", "Filter Plant",
                                 value=dcc.Dropdown(id='filter_plant',
                                            options= [{'label': i , 'value': i } for i in pl_id_list+['ALL']],
                                            value='ALL',
                                            placeholder="Select a plant",
                                            multi=True
                                            ))]),
            
            dbc.Col([create_card("filter_warehouse", "Filter Warehouse",
                                 value=dcc.Dropdown(id='filter_warehouse',
                                            options= [{'label': i , 'value': i } for i in wr_id_list+['ALL']],
                                            value='ALL',
                                            placeholder="Select a warehouse",
                                            multi=True
                                            ))]),
            
            dbc.Col([create_card("filter_volume", "Filter Volume",
                                 value=dcc.RangeSlider(id='filter_volume',
                                            min=0, max=5_000, step=300,
                                           value=[min_supply, max_supply]))])
            
            ], style={"height": "5%"}),
        
        dbc.Row([
                dbc.Col([create_card("map", "Supply Chain Network",
                                     value=dcc.Graph(id='map'))])
                ]),
        dbc.Row([
                dbc.Col([create_card("warehouse_pie", "Demand",
                                     value=dcc.Graph(id='warehouse_pie'))],
                        width={"size": 5, "order": "first", "offset": 0}),
                dbc.Col([create_card("plant_pie", "Supply",
                                     value=dcc.Graph(id='plant_pie'))],
                        width={"size": 5, "order": "first", "offset": 0})
                ]),
        dbc.Row([
                dbc.Col([create_card("waterfall_costs", "Costs",
                         value=dcc.Graph(id='waterfall_costs'))],
                         width={"size": 5, "order": "first", "offset": 0})
                        ,
                dbc.Col([create_card("supply_demand", "Supply - Demand",
                                     value=dcc.Graph(id='supply_demand'))],
                        width={"size": 5, "order": "first", "offset": 0})
                ])
    ]
)

def get_costs(filtered_df):
    
    trp_cost = np.round(filtered_df['transport_cost'].sum())
    op_cost = np.round(filtered_df['operate_cost'].sum())
    total_supply = np.round(filtered_df['supply'].sum())
    
    return trp_cost, op_cost, total_supply

def get_pies(filtered_df):
    
    if filtered_df.shape[0] > 0:
        pie_supply = filtered_df[['pl_id', 'supply']].groupby('pl_id').sum().reset_index()
    else:
        pie_supply = pd.DataFrame({'pl_id':[],
                                    'supply':[]
                                    })
    
    
    fig_supply = px.pie(pie_supply, values='supply', 
    names='pl_id', 
    title=f'')
    
    #print(filtered_df)
    if filtered_df.shape[0] > 0:
        pie_demand = filtered_df[['wr_id', 'demand']].groupby('wr_id').max().reset_index()
    else:
        pie_demand = pd.DataFrame({'wr_id':[],
                                    'demand':[]
                                    })
    
    fig_demand = px.pie(pie_demand, values='demand', 
    names='wr_id', 
    title=f'')
    
    return fig_supply, fig_demand


def get_waterfall(filtered_df, trp_cost, op_cost):
    
    #waterfall
    fig_waterfall = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative", "relative", "total"],
        x = ["Transportation", "Operations", "Total"],
        textposition = "outside",
        y = [trp_cost, op_cost, trp_cost+op_cost],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    
    if filtered_df.shape[0] > 0:
        bar_pl = pd.DataFrame({'supply/demand': ['supply','demand'],
                               'value': [
                                   filtered_df['supply'].sum(),
                                   filtered_df[['wr_id', 'demand']].groupby('wr_id').max().reset_index()['demand'].sum()
                                   ]})
    else:
        bar_pl = pd.DataFrame({'supply/demand':[],
                                    'value':[]
                                    })
    
    fig_supply_demand = px.bar(bar_pl,
                           x='supply/demand', y='value')
    
    return fig_waterfall, fig_supply_demand

def get_map(filtered_df):
    fig = go.Figure()
    
    closed_plants = list(set(plant_loc['pl_id']) - set(filtered_df['pl_id']))
    closed_plants_df = plant_loc[plant_loc['pl_id'].isin(closed_plants)]
    
    # Add locations plants
    fig.add_trace(go.Scattergeo(
        lon=filtered_df['lon_pl'],
        lat=filtered_df['lat_pl'],
        text='',
        marker=dict(size=[8 for x in range(filtered_df.shape[0])],
                    symbol=['circle' for x in range(filtered_df.shape[0])],
                    color=['blue' for x in range(filtered_df.shape[0])],
                    line=dict(width=3, color='rgba(68, 68, 68, 0)')
                    )))
    
    # Add locations closed plants
    fig.add_trace(go.Scattergeo(
        lon=closed_plants_df['lon_pl'],
        lat=closed_plants_df['lat_pl'],
        text='',
        marker=dict(size=[8 for x in range(closed_plants_df.shape[0])],
                    symbol=['circle' for x in range(closed_plants_df.shape[0])],
                    color=['red' for x in range(closed_plants_df.shape[0])],
                    line=dict(width=3, color='rgba(68, 68, 68, 0)')
                    )))
    
    # Add locations warehouses
    fig.add_trace(go.Scattergeo(
        lon=filtered_df['lon_wr'],
        lat=filtered_df['lat_wr'],
        text='',
        marker=dict(size=[8 for x in range(filtered_df.shape[0])],
                    symbol=['square' for x in range(filtered_df.shape[0])],
                    color=['black' for x in range(filtered_df.shape[0])],
                    line=dict(width=3, color='rgba(68, 68, 68, 0)')
                    )))
    
    for index, record in filtered_df.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[record['lon_pl'], record['lon_wr']],
            lat=[record['lat_pl'], record['lat_wr']],
            mode='lines',
            line=dict(width=2, color='green'),
            opacity=0.8,
            showlegend=False,
        ))
        
    # Specify the layout attributes
    title = ""
    layout = dict(title=title,
                  showlegend=False,
                  geo=dict(
                      projection_scale=5, #this is kind of like zoom
                      center=dict(lat=coord_init[0], lon=coord_init[1]),
                      showland=True,
                      landcolor='rgb(243, 243, 243)',
                      countrycolor='rgb(204, 204, 204)'))
    
    fig.update_layout(layout)
    fig.update_layout(height=800) 
        
    return fig
    

@app.callback([Output(component_id='C_cost_card', component_property='children'),
               Output(component_id='c_cost_card', component_property='children'),
               Output(component_id='volume_card', component_property='children'),
               Output(component_id='warehouse_pie', component_property='figure'),
               Output(component_id='plant_pie', component_property='figure'),
               Output(component_id='waterfall_costs', component_property='figure'),
               Output(component_id='supply_demand', component_property='figure'),
               Output(component_id='map', component_property='figure')],
              [Input(component_id='filter_plant', component_property='value'),
               Input(component_id='filter_warehouse', component_property='value'),
               Input(component_id='filter_volume', component_property='value')])

def get_output(pl_id, wr_id, vol):
    #print(pl_id)
    pl_id_filt = pl_id_list if ((pl_id == 'ALL') or (len(pl_id) == 0))  else pl_id
    wr_id_filt = wr_id_list if ((wr_id == 'ALL') or (len(wr_id)==0)) else wr_id
    
    #print(pl_id_filt)
    
    filtered_df = report[(report['pl_id'].isin(pl_id_filt)) & 
                         (report['wr_id'].isin(wr_id_filt)) &
                         (report['supply'] >=  vol[0]) & (report['supply'] <=  vol[1])].copy()
    
    #print(report)
    #print(vol[1])
    #print(filtered_df)
    #add coord
    filtered_df = filtered_df.merge(plant_loc, left_on='pl_id', right_on='pl_id')
    filtered_df = filtered_df.merge(warehouse_loc, left_on='wr_id', right_on='wr_id')
    
    
    trp_cost, op_cost, total_supply = get_costs(filtered_df)
    #pies
    fig_supply, fig_demand = get_pies(filtered_df)

    fig_waterfall, fig_supply_demand = get_waterfall(filtered_df, trp_cost, op_cost)
    
    sc_map = get_map(filtered_df)
    
    return(str(int(trp_cost)), str(int(op_cost)), str(int(total_supply)),
             fig_demand, fig_supply, fig_waterfall, fig_supply_demand,
             sc_map)




# Run the app
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)
