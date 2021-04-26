import dash
import dash_core_components as dcc
import dash_html_components as html
#import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from joblib import dump, load
import pickle
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve, precision_score, auc, f1_score
import dash_table
import numpy as np
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
#from IPython.display import HTML
import plotly.io as pio


gb_top_features = pd.read_csv('lgb_imp_feat.csv')
rf_top_features = pd.read_csv('rf_imp_feat.csv')
df_gb = pd.read_csv('Comp_Table_SVM.csv')
df_rf = pd.read_csv('Comp_Table_RF.csv')
df_lr = pd.read_csv('Comp_Table_lr.csv')



pio.renderers.default = "notebook"

ufc_data = pd.read_csv("CleanedMerged.csv")
# print(ufc_data.head(5))

#ufc_data['date_year'] = ufc_data['date_year'].apply(lambda x: x.year)
########### viz 1 #################
values = ufc_data['date_year'].sort_values(ascending=False).value_counts().sort_index()
labels = values.index

colors = ['lightslategray', ] * 27
colors[21] = 'crimson'

trace_event = go.Bar(
    x=labels, y=values, name= 'UFC Event Per Year',

    marker_color=colors  # marker color can be a single color value or an iterable
)

data = [trace_event]

layout = dict(title = "UFC Event Per Year",
              showlegend = False)

fig = dict(data=data, layout=layout)
########### viz 1 #################
########### viz 2 #################

values = ufc_data.groupby(['country']).count()['R_fighter'].values
labels = ufc_data.groupby(['country']).count()['R_fighter'].index


colors = ['lightslategray', ] * 27
colors[21] = 'crimson'

trace_country = go.Bar(
    x=labels, y=values, name= 'UFC Event by Country',

    marker_color=colors  # marker color can be a single color value or an iterable
)

data1 = [trace_country]

layout1 = dict(title = "UFC Event by Country",
              showlegend = False)

fig1 = dict(data=data1, layout=layout1)

########### viz 2 #################
########### viz 3 #################

# female_fighters = ufc_data.weight_class.str.contains('Women').sum()
# male_fighters = len(ufc_data['weight_class'])-female_fighters
#
labels = ['female','male']
female_fighters = ufc_data.weight_class.str.contains('Women').sum()
male_fighters = len(ufc_data['weight_class'])-female_fighters
values = female_fighters, male_fighters

trace_gender = go.Pie(labels=labels, values=values,
               textfont=dict(size=20)
               )
# Fill out the data wtih traces
data2 = [trace_gender]
# Create one annotation to each day (Monday, Tuesday and so on)
ann2 = dict(font=dict(size=20),
            # Specify text position (place text in a hole of pie)
            x=0.23,
            y=0.5
            )
layout2 = go.Layout(title ='Number of fighters by gender',
                   annotations=[ann2])

fig2 = dict(data=data2, layout=layout2)
########### viz 3 #################
########### viz 4 #################
values=ufc_data['Winner'].value_counts()[:10].values
labels=ufc_data['Winner'].value_counts()[:10].index

trace_winner = go.Pie(labels=labels, values=values,
               textfont=dict(size=20)
               )
# Fill out the data wtih traces
data3 = [trace_winner]
# Create one annotation to each day (Monday, Tuesday and so on)
ann1 = dict(font=dict(size=20),
            # Specify text position (place text in a hole of pie)
            x=0.23,
            y=0.5
            )
layout3 = go.Layout(title ='Win Distribution',
                   annotations=[ann1])

fig3 = dict(data=data3, layout=layout3)
########### viz 5 #################

ufc_data['R_age'] = ufc_data['R_age'].fillna(ufc_data['R_age'].median())
value1 = ufc_data.groupby('R_age').count()['R_fighter'].values
labell = ufc_data.groupby('R_age').count()['R_fighter'].index

ufc_data['B_age'] = ufc_data['B_age'].fillna(ufc_data['B_age'].median())
value2 = ufc_data.groupby('B_age').count()['R_fighter'].values
label2 = ufc_data.groupby('B_age').count()['R_fighter'].index

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig4 = make_subplots(rows=1, cols=2)

fig4.add_trace(
    go.Bar(y=value2,x=label2),
    row=1, col=2
)

fig4.add_trace(
    go.Bar(y=value1,x=labell),
    row=1, col=1
)

fig4.update_layout(height=500, width=900,
                  title_text="Distribution of Age for both the team fighters",
                  xaxis=dict(title='Age'), yaxis=dict(title='Count'))


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
	html.Header([
           		html.H1("IST 707 Final Project Dashboard", style={'height': '70px','width':'95%', 'display': 'inline-block','textAlign':'center'}),
                html.Img(src="/assets/syrlogo2.jpg", style={'height': '70px','width':'5%', 'display': 'inline-block','textAlign': 'right'})],
                style={'backgroundColor':'orange','height': '70px'}, className="banner"),
    dcc.Tabs([
    	dcc.Tab(label='About', children=[
    		dcc.Markdown('''
			### UFC Fight
			#### Introduction:
			**ISports betting is a $155 billion industry. Fighting ranks among the top in the industry, and the Ultimate Fighting Championship (UFC) is currently taking steps to push it even further. Mixed Martial Arts (MMA) fighter statistics involve everything from skill centric values such as wins, and significant strikes landed to physiological measurements such as height and reach. There are over one hundred different features up to analyze before any given fight, and machine learning can be used to best understand which are most relevant, and to indent trends and predict the outcomes (win/draw/loss) of each fight.**


			#### Problem:
			**The goal of this study is to explore our ability to predict the outcome of UFC fights based on each match’s pre-fight statistics using machine learning models. An accurate prediction model could both inform the best placed bets (and potential risk associated) for each fight, but also could provide insight to coaches when accepting fights to begin with, simply by looking at the opponent’s statistics relative to their fighter. It could also be used to help to identify which features are most significant in this prediction.**

			#### Creators:
			- **Saheb Singh**
			- **Laxman Kumar**
			- **Abhiraj Singh**
			'''
			)
    		]),
    	dcc.Tab(label='Exploratory Data Analysis', children=[
            html.H1(children='Exploratory Data Analysis'),
            html.P('Visualization of the target variable. The dataset is quite imbalanced as can be seen below'),
            dcc.Graph(id="UFC Event Per Year", style={"width": "75%", "display": "inline-block"},
                figure=fig ),
            dcc.Graph(id="UFC Event by Country", style={"width": "75%", "display": "inline-block"},
                figure= fig1),
            dcc.Graph(id="Number of fighters by gender", style={"width": "75%", "display": "inline-block"},
                figure= fig2),
            dcc.Graph(id="Win Distribution", style={"width": "75%", "display": "inline-block"},
                figure= fig3),
            dcc.Graph(id="Distribution of Age for both the team fighters", style={"width": "75%", "display": "inline-block"},
                figure= fig4)
        ]),
        
        ### Random Forest 
        dcc.Tab(label='Random Forest', children=[
            html.H1(children='Random Forest'),
            html.P('Select Number of Estimators'),
            dcc.Dropdown(
                id='RF_Estimators',
                options=[{'label': i, 'value': i} for i in ['100','200','400']],
                value=200,
                style={'width': '160px'}
            ),
            html.P('Select Max Depth'),
            dcc.Dropdown(
                id='RF_Max_Depth',
                options=[{'label': i, 'value': i} for i in [50,10]],
                value= 50,
                style={'width': '160px'}
            ),
            html.P('Select Min Samples Split'),
            dcc.Dropdown(
                id='RF_Min_Samples_Split',
                options=[{'label': i, 'value': i} for i in [100,50,30]],
                value= 100,
                style={'width': '160px'}
            ),
            html.P('Select Min Samples Leaf'),
            dcc.Dropdown(
                id='RF_Min_Samples_Leaf',
                options=[{'label': i, 'value': i} for i in [1,2,4]],
                value= 1,
                style={'width': '160px'}
            ),
            html.Div(id='rf_evaluation'),
            dcc.Graph(id='rf-features')
        ]),
            
        ### Support Vector Machine
        dcc.Tab(label='Support Vector Machine', children=[
            html.H1(children='Support Vector Machine'),
            html.P('Select Lambda Regularization Parameter'),
            dcc.Dropdown(
                id='SVM_C',
                options=[{'label': i, 'value': i} for i in [0.001,0.01,0.1,1.0,10,100,1000]],
                value=0.01,
                style={'width': '160px'}
            ),
            html.P('Select Penalty'),
            dcc.Dropdown(
                id='Penalty',
                options=[{'label': i, 'value': i} for i in ['l1','l2']],
                value= 'l2',
                style={'width': '160px'}
            ), 
            html.Div(id='svm_evaluation'),
            dcc.Graph(id='svm_features')
        ]),
            
        ### Logistic Regression
        dcc.Tab(label='Logistic Regression', children=[
            html.H1(children='Logistic Regression'),
            html.P('Select C'),
            dcc.Dropdown(
                id='LR_C',
                options=[{'label': i, 'value': i} for i in [1,0.01,0.1]],
                value=0.01,
                style={'width': '160px'}
            ),
            html.P('Select Iteration'),
            dcc.Dropdown(
                id='LR_max_iter',
                options=[{'label': i, 'value': i} for i in [100,200,400]],
                value= 100,
                style={'width': '160px'}
            ),
            html.Div(id='lr_evaluation')
        ]),
    ])
])
@app.callback([Output('rf_evaluation', 'children'),
               Output('rf-features', 'figure')],
              [Input('RF_Estimators', 'value'),
               Input('RF_Max_Depth', 'value'),
               Input('RF_Min_Samples_Split', 'value'),
               Input('RF_Min_Samples_Leaf', 'value')
               ]
)
def rand_for(n_estimators,max_depth,min_samples_split,min_samples_leaf):
    tab = df_rf[(df_rf['max_depth']==max_depth) & (df_rf['min_samples_split']==min_samples_split) & (df_rf['min_samples_leaf']==min_samples_leaf) & (df_rf['n_estimators']==n_estimators)]
    tab.drop(['max_depth','min_samples_leaf','min_samples_split','n_estimators'],axis = 1,inplace = True)
    
    fig = px.bar(rf_top_features, x="index",y="Importance",title="Important Feature")
     
    return [html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in tab.columns],
                data=tab.to_dict("rows"),
                style_cell={'width': '300px',
                'height': '60px',
                'textAlign': 'left'})
            ]),fig]

@app.callback([Output('svm_evaluation', 'children'),
                Output('svm_features', 'figure')],
              [Input('SVM_C', 'value'),
              Input('Penalty', 'value')]
)
def grad_boost(SVM_C,Penalty):

    tab = df_gb[(df_gb['C']==SVM_C) & (df_gb['Penalty']==Penalty)]
    fpr = df_gb['fpr']
    tpr = df_gb['tpr']

    tab.drop(['C','Penalty','fpr','tpr'],axis = 1,inplace = True)

    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr}, columns=['fpr', 'tpr'])
    fig = px.line(df_roc, x='fpr', y='tpr', title="TPR vs FPR")

    #fig = px.line(df_roc, x="fpr", y="tpr", title='ROC Curve')

    return [html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in tab.columns],
                data=tab.to_dict("rows"),
                style_cell={'width': '300px',
                'height': '60px',
                'textAlign': 'left'})
            ]),fig]

@app.callback([Output('lr_evaluation', 'children')],
              [Input('LR_C', 'value'),
              Input('LR_max_iter', 'value')]
)
def log_reg(C,max_iter):
    tab = df_lr[(df_lr['C']==C) & (df_lr['max_iter']==max_iter)]
    index = tab['Number']
    tab.drop(['C','max_iter','Number'],axis = 1,inplace = True)
    #tpr1 = tpr[index]
    #fpr1 = fpr[index]    

    #print(type(fpr))
    
    #df_roc = pd.DataFrame({'fpr': fpr1, 'tpr': tpr1}, columns=['fpr', 'tpr'])
    
    #fig = px.scatter(df_roc, x="fpr", y="tpr", title='ROC Curve')
    return [html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in tab.columns],
                data=tab.to_dict("rows"),
                style_cell={'width': '300px',
                'height': '60px',
                'textAlign': 'left'})
            ])]

if __name__ == '__main__':
 app.run_server(host = '127.0.0.1')