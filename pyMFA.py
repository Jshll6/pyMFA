#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:11:23 2023

@author: Johanne Schjødt-Hansen
"""
import pandas as pd 
import numpy as np
from scipy.stats import norm 
import os
import yaml
import warnings

class Model:
    def __init__(self,
                 config):
        cwd = os.getcwd()
        filepath = os.path.join(cwd,config["input"]["filename"])
        sheets = config["input"]["sheet_names"]
        cols = config["input"]["columns"]
        self.output = config["output"]
        self.cols = cols
        
        # Load excelfile and relevant sheets
        xls = pd.ExcelFile(filepath) 
        self.system = pd.read_excel(xls,sheet_name=sheets["system"])

        self.parameters = pd.read_excel(xls,
                                        sheet_name=sheets["parameters"],
                                        index_col=cols["prod"])
        self.production = pd.read_excel(xls,
                                        sheet_name=sheets["hist"],
                                        index_col=cols["prod"])
        self.tc = pd.read_excel(xls,
                                sheet_name=sheets["tc_values"],
                                index_col=cols["flow"])
        self.partitioning = pd.read_excel(xls,
                                          sheet_name=sheets["partitioning"],
                                          index_col=cols["prod"])

        # ignore warning from data validation/drop down meny in excel
        warnings.simplefilter(action='ignore', category=UserWarning)
        
        # Load model start and end year
        self.start = self.system[cols["start"]]
        self.end = self.system[cols["end"]]
        
        # Model name and flow unit
        self.name = self.system[cols["name"]]
        self.unit = self.system[cols["unit"]]
        
        # Load list of materials/products
        self.materials = self.production.index 
        # Load list of processes and defining 
        self.processes = pd.concat([self.tc[cols["from"]],
                                         self.tc[cols["to"]]]).unique()
        self.stock_process = cols["stock_process"]
        self.rec_process = cols["recycling_process"]
        self.post_rec_flow = cols["recycled_flow"]
        self.pre_rec_flow = cols["pre_recycled_flow"]
        
        # Defining modeling/historic timesteps and timestep 0
        self.timesteps = range(int(self.start),
                               int(self.end)+1)
        self.timestep_0 = int(self.system[cols["start"]]-1)
        self.timesteps_hist = range(int(self.production.columns[0]),
                                    self.timestep_0+1)

        # Load import flows 
        self.import_flows = self.parameters[cols["import"]]
        
        # Load in-use parameters of materials/products
        self.lifetime = self.parameters[cols["lifetime"]]
        self.sd = self.parameters[cols["sd"]]
        self.q = cols["q_fraction"]
        self.g_rate = self.parameters[cols["g_rate"]]
        self.stock_0 = self.parameters[cols["stock"]]
        self.growth_type = cols["growth_type"]
        
        # Load sankey statements
        self.sankey_statement = config["output"]["generate_sankey"]
        self.sankey_years = config["output"]["sankey_years"]
    
    def calculate_production(self): # additions: fixed_adjustable,production_previous,g_rate
        # Input to system (production of material)
    
        #if fixed_adjustable == "fixed":
        #    production = pd.DataFrame(columns=[self.timesteps]) 
        #    production_previous = self.production[self.timestep_0]
        #    production[timestep] = production_previous*(1+self.g_rate)**(timestep-self.timestep_0) 
        #else:
        #    production_previous = production_previous
        #    production[timestep] = production_previous*(1+g_rate)
            
        production = pd.DataFrame() 
        if self.growth_type == "exponential":
            # Exponential growth
            for timestep in self.timesteps:
                production[timestep] = self.production[self.timestep_0]\
                                        *(1+self.g_rate)**(timestep-self.timestep_0) 
        if self.growth_type == "static":
            production = self.production[self.timestep_0]
        #else:
        #    print("Add growth type calculation to model")
        return production
    
    def stock_timesteps(self):
        stock_timesteps = np.ceil(norm.ppf(self.q,
                                       self.lifetime,
                                       self.sd)+1)
        return stock_timesteps
    
    def stock_release_fraction(self):
        cdfs = []
        f_outs = []
        X = self.stock_timesteps()
        for material in range(len(self.materials)):
            cdf = norm.cdf(range(int(X[material])),
                           self.lifetime[material],
                           self.sd[material])
            f_out = np.append(cdf[0],np.diff(cdf))
            cdfs.append(cdf)
            f_outs.append(f_out)
            # Calculate stock in year 0 (if time)
            if np.isnan(self.stock_0[material]) == True:
                print("stock value not provided for material")
                # add stock calculation if time
                self.stock_0[material]=0
            else:
                pass
        return f_outs
    
    def historic_stock_release(self):
        # ac: age_cohorts (mass flows out of stock over time from each input flow)
        ac = [] 
        f_outs = self.stock_release_fraction()
        for timesteps in self.timesteps_hist:
            # calculate mass flows in age cohorts from stock in timesteps prior to model
            ac_t = self.production[timesteps].multiply(f_outs)
            ac.append(ac_t)
        
        ac = pd.concat(ac,axis=1)
        return ac
    
    def calculate_flows(self,
                        df_tc, # dataframe with flows 
                        start_process_input,
                        processes,
                        flows_prior,
                        flows_to):
        I = start_process_input.transpose()
        df = df_tc
        # array of processe
        unique_tot = processes

        # Create separate DataFrames for each unique value in 'From'
        dfs_per_from = {}
        dfs_per_to = flows_to
        dfs_from = flows_prior
        #dfs_from = pd.DataFrame(columns=df.columns)

        for p in range(len(unique_tot)-1):
            # Filter rows for the current 'From' value
            filter_from = df[df[self.cols["from"]] == unique_tot[p]].copy() 
            # Multiply 'tc' columns by the input value
            filter_from[filter_from.select_dtypes(include=['number']).columns]\
                            = filter_from[filter_from.select_dtypes(include=['number'])\
                            .columns]\
                            .multiply(I.values) 
                            
            dfs_from=pd.concat([dfs_from,filter_from])
            
            filter_to = dfs_from[dfs_from[self.cols["to"]] == unique_tot[p+1]]
            I = pd.DataFrame(filter_to[filter_to.select_dtypes(include=['number'])\
                                       .columns].sum())
            I = I.transpose()
            dfs_per_from[unique_tot[p]] = filter_from
            dfs_per_to[unique_tot[p+1]] = filter_to
            
        return dfs_from,dfs_per_to

    def dynamic_stock_model(self,
                            timestep,
                            input_stock_process,
                            age_cohorts,
                            d_stocks,
                            stocks):
        # Mass flows in  all age-cohorts out of use from input in timestep
        ac = age_cohorts 
        #d_stocks = pd.DataFrame(np.nan,index=self.production.index,columns=[int(self.start.values)])
        d_stocks = d_stocks
        X = self.stock_timesteps()
        f_outs = self.stock_release_fraction()
        # Dynamic stock model
        I = input_stock_process
        # add age cohorts outflows from current timestep to previous years
        ac[timestep]=I.multiply(f_outs) 
        
        # create storage for delta stock
        d_stock = []
        flows_t = []
        
        for material in range(len(self.materials)):
            flows_ac = [] # mass out in timestep
            for age_cohort in np.flip(range(int(timestep-X[material]+1),int(timestep+1))):
                # mass flow from age cohort 
                flow_ac = ac[age_cohort][material][int(timestep-age_cohort)]
                # gather mass flow from each age cohort
                flows_ac.append(flow_ac) 
            # flow out of stock process in timestep, t, is sum of mass flows from all age cohorts
            flow_m = sum(flows_ac)
            # delta stcok is the change in input vs. output
            d_stock_sum = I[material]-sum(flows_ac) 
            d_stock.append(d_stock_sum) 
            flows_t.append(flow_m)
        
        d_stocks[timestep]=d_stock
            
        stocks[timestep] = stocks[int(timestep-1)]+d_stocks[timestep]
        #tc = self.tc[self.tc[self.cols["from"]].str.contains(process)]
        #flows = self.d_stocks[timestep]*
        return d_stocks,stocks,flows_t
    
    def calculate_mfa_system(self):
        # Create dataframe for storing stocks in dynamic stock model
        stocks_0 = pd.DataFrame(self.stock_0) 
        # Change column name to timestep 0 value
        stocks_0.columns = [self.timestep_0]
        # calculate age-cohorts
        age_cohorts = self.historic_stock_release()
        # initiate delta stock 
        d_stock = pd.DataFrame(index=self.materials,columns=[int(self.start.values)]) 

        # Initializing recycling flows
        df_pre_rec = pd.DataFrame(columns=self.timesteps,index=self.materials)
        df_post_rec = pd.DataFrame(columns=self.timesteps,index=self.materials)
        df_tot_rec = pd.DataFrame(columns=self.timesteps,index=self.materials)
        
        # Initializing dataframe for fixed import flow
        df_import = pd.DataFrame(self.import_flows)
        df_import.columns=[int(self.start)]
        # Initializing dataframe for reycling flows (no recycling in year 0)
        rec_flows = pd.DataFrame(columns=self.timesteps,index=self.materials)
        # calculate production in modeling timesteps
        production = self.calculate_production()
        # initializing flows
        flows_all = []
        df_prod_virgin = pd.DataFrame(columns=self.timesteps,index=self.materials)
        # list of processes before stock
        p_before_stock = self.processes \
                            [0:list(self.processes).index(self.cols['stock_process'])+1]

        for timestep in self.timesteps:                   
            # calculate flows prior to stock process - including input to stock process
            flows_before,flows_to_before = self.calculate_flows(df_tc = self.tc,
                                                                start_process_input = production[timestep],
                                                                processes = p_before_stock,
                                                                flows_prior = pd.DataFrame(columns=self.tc.columns),
                                                                flows_to = {})
            # pre-consumer recycling 
            pre_rec = flows_before[flows_before[self.cols["to"]] == self.pre_rec_flow]
            # Add pre-consumer recycling flows for timestep to dataframe
            df_pre_rec[timestep] = pre_rec[pre_rec.select_dtypes(include=['number']).columns].transpose().values
            # define input to stock process
            input_stock_process = flows_to_before[self.cols['stock_process']]\
                                    [flows_to_before[self.cols['stock_process']]\
                                     .select_dtypes(include=['number']).columns]\
                                        .transpose().squeeze()
            # calculate dynamic stock model
            d_stocks,stocks,flows_t = self.dynamic_stock_model(timestep,
                                                                  input_stock_process,
                                                                  age_cohorts,
                                                                  d_stock,
                                                                  stocks_0)
            # create list of flows after stock process - must include stock process 
            p_after_stock = self.processes[list(self.processes).index(self.cols['stock_process']):]
            
            # calcualate remaining system flows
            flows,flows_to = self.calculate_flows(df_tc = self.tc,
                                                  start_process_input = pd.Series(flows_t),
                                                  processes = p_after_stock,
                                                  flows_prior = flows_before,
                                                  flows_to = flows_to_before)
            
            flows_all.append(flows)
            # Post consumer recycling 
            post_rec = flows[flows[self.cols["to"]] == self.post_rec_flow]
            # Add post-consumer recycling flows for timestep to dataframe
            df_post_rec[timestep]=post_rec[post_rec.select_dtypes(include=['number']).columns].transpose().values
            # Total recycled material
            df_tot_rec[timestep]=df_pre_rec[timestep].values+df_post_rec[timestep].values
            # Recycling flows for next timestep (multiplied with partitioning)
            rec_flow_matrix = df_tot_rec[timestep].values*self.partitioning
            rec_flows[timestep+1] = rec_flow_matrix.sum(axis=1)
            # Amount of production from virgin input
            df_import[timestep]=self.import_flows.values
            production_virgin = production[timestep]-rec_flows[timestep]-df_import[timestep]
            df_prod_virgin[timestep] = production_virgin
            # Add sum column to flows dataframe
            
        #df_pre_rec = df_pre_rec.drop([self.timestep_0],axis=1)
        #df_post_rec = df_post_rec.drop([self.timestep_0],axis=1)
        #df_tot_rec = df_tot_rec.drop([self.timestep_0],axis=1)
        stocks = stocks.drop([self.timestep_0],axis=1) 
        
        return flows_all,flows_to,d_stocks,stocks,rec_flows,df_pre_rec,df_post_rec,df_prod_virgin,df_import,production

    def sankey_data(self,year,year_index):
        # calculate mfa system
        flows_all,flows_to,d_stocks,stocks,rec_flows,pre_rec,post_rec,prod_virgin,df_import,production = self.calculate_mfa_system()
        # input values
        year = year 
        year_index = year_index
        # defining virgin flow dateframe
        virgin_flow= pd.DataFrame([prod_virgin[year].values.tolist()],columns=self.materials)
        virgin_flow[self.cols["from"]]="Virgin"
        virgin_flow[self.cols["to"]]="Manufacturing"

        # defining recycling flow dataframe 
        recycling_flow= pd.DataFrame([rec_flows[year].values.tolist()],columns=self.materials)
        recycling_flow[self.cols["from"]]="Recycled"
        recycling_flow[self.cols["to"]]="Manufacturing"

        # defining import flow dataframe
        import_flow= pd.DataFrame([df_import[year].values.tolist()],columns=self.materials)
        import_flow[self.cols["from"]]="Import"
        import_flow[self.cols["to"]]="Manufacturing"
        # defining delta stock dataframe
        dstock_flow= pd.DataFrame([d_stocks[year].values.tolist()],columns=self.materials)
        dstock_flow[self.cols["from"]]="Use"
        dstock_flow[self.cols["to"]]="Delta stock"
        # Removing production row from flows dataframe
        flows_without_prod = flows_all[year_index].drop([1])
        
        # Creating new flows dataframe containing all nodes for the sankey diagram
        df_flows = pd.concat([virgin_flow,recycling_flow,import_flow,flows_without_prod])
        # Inserting delta stock at relevant location
        df_flows.loc[6.5]=dstock_flow.loc[0]
        df_flows = df_flows.sort_index().reset_index(drop=True)
        df_flows.drop(df_flows.loc[df_flows['To process']=='Final disposal'].index, inplace=True)
        
        df_melt = df_flows.melt(id_vars=["From process", "To process"],
                                var_name="Material",
                                value_name="Values")
        
        processes = pd.concat([df_flows[self.cols["from"]],
                                         df_flows[self.cols["to"]]]).unique()
        processes_df = pd.DataFrame(processes)
        #sources = []
        #targets = []
        
        df_flows['Sources']=df_flows[self.cols["from"]]\
                                            .replace(processes,processes_df.index)
        df_flows['Targets']=df_flows[self.cols["to"]]\
                                            .replace(processes,processes_df.index)
        sources = df_flows['Sources']
        targets = df_flows['Targets']
        
        df_melt['Sources']=df_melt[self.cols["from"]]\
                                            .replace(processes,processes_df.index+1)
        df_melt['Targets']=df_melt[self.cols["to"]]\
                                            .replace(processes,processes_df.index+1)
        sources = df_melt['Sources']
        targets = df_melt['Targets']
        values = df_melt['Values']
        
        link_colors=["rgba(232, 63, 72, 0.8)"]*len(df_flows)+\
                       ["rgba(47, 62, 234, 0.8)"]*len(df_flows)+\
                       ["rgba(0, 136, 53, 0.5)"]*len(df_flows)+\
                       ["rgba(121, 35, 142, 0.8)"]*len(df_flows)+\
                       ["rgba(246, 208, 77, 0.8)"]*len(df_flows)+\
                       ["rgba(252, 118, 52, 0.8)"]*len(df_flows)+\
                       ["rgba(47, 62, 234, 0.5)"]*len(df_flows)
        
        return sources, targets, values, link_colors
      
    def sankey_diagram(self,filename_single,filename_double):
        import plotly.graph_objects as go
        import plotly.io as pio


        legend_colors = ["rgba(255,255,255,0)",
                            "rgba(232, 63, 72, 0.8)",
                            "rgba(47, 62, 234, 0.8)",
                            "rgba(0, 136, 53, 0.5)",
                            "rgba(121, 35, 142, 0.8)",
                            "rgba(246, 208, 77, 0.8)",
                            "rgba(252, 118, 52, 0.8)",
                            "rgba(47, 62, 234, 0.5)",
                            "rgba(255,255,255,0)",
                            "rgba(80,80,80,1)",
                            "rgba(0,136,52,1)",
                            "rgba(153,0,0,1)",
                            "rgba(0,0,0,1)"]
        legend_names = ['Flows:',
                        'Food packaging 2d', 
                        'Food packaging bottles', 
                        'Food packaging other',
                        'Non-food packaging 2d', 
                        'Non-food packaging bottles',
                        'Non-food packaging other', 
                        'Fibers',
                        'Processes:',
                        'System process',
                        'Recycled material',
                        'Final disposal', 
                        'Leaving system']
        
        labels = ['',
                  'Virgin','Recycled','Import',
                  'Manufacturing',
                  'Use',
                  'Source seperated','Residual',
                  'Sorting',
                  'Reprocessing',
                  'Pre consumer recycled',
                  'Export ',
                  'Delta stock',
                  'Landfill','Incineration',
                  'Post consumer recycled']
        node_colors = ["rgba(80,80,80,1)",
                       "rgba(80,80,80,1)","rgba(80,80,80,1)","rgba(80,80,80,1)",
                        "rgba(80,80,80,1)",
                        "rgba(80,80,80,1)",
                        "rgba(80,80,80,1)","rgba(80,80,80,1)",
                        "rgba(80,80,80,1)",
                        "rgba(80,80,80,1)",
                        "rgba(0,136,52,1)",
                        "rgba(0,0,0,1)",
                        "rgba(80,80,80,1)",
                        "rgba(153,0,0,1)","rgba(153,0,0,1)",
                        "rgba(0,136,52,1)"]
        x_values = [0,
                    0,0,0,
                    0.17,
                    0.33,
                    0.5,0.5,
                    0.67,
                    0.83,
                    0.95,
                    0.67,
                    0.37,
                    0.9,0.9,
                    0.95]

        y_values = [0]*len(labels)
        
        # Single sankey diagram
        year = 2030
        year_index = 11
        sources,targets,values,link_colors = self.sankey_data(year, year_index)
        if self.sankey_statement == 'yes':
            print('Generating sankey')
            fig_single = go.Figure(go.Sankey(
                #arrangement = "snap",
                valuesuffix = self.unit[0],
                node = dict(
                    label = labels, 
                    pad = 10,
                    x = x_values,
                    y = y_values,
                    color = node_colors
                ),
                domain={
                'x': [0,0.9]
                }, 
                link = {
                    "source": sources,
                    "target": targets,
                    "value" : values,
                    "color" : link_colors
                    
                    }
                ))
            fig_single.update_layout(title_text=self.name[0]+' '+str(year),
                  font_size=20)
            #Set default renderer for graphic to browser
            pio.renderers.default = "browser"
            #Show figure with renderer
            #Save figure as website
            fig_single.show
            fig_single.write_html(filename_single),
            # Double sankey diagram
            # Generate sankey data for first subplot 
            sources_1,targets_1,values_1,link_colors_1 = self.sankey_data(2030, 11)
            # Generate sankey data for second subplot 
            sources_2,targets_2,values_2,link_colors_2 = self.sankey_data(2020, 1)
            trace1 = go.Sankey(valuesuffix = self.unit[0],
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(
                        color = "black",
                        width = 0.5
                    ),
                    x = x_values,
                    y = y_values,
                    label = labels,
                    color = node_colors
                ),
                link = dict(
                    source = sources_1,
                    target = targets_1,
                    value = values_1,
                    color = link_colors_1
                ),
                domain={
                    'y': [0,0.6],
                    'x': [0,0.9] 
                }
            )
            trace2 = go.Sankey( valuesuffix = self.unit[0],
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(
                        color = "black",
                        width = 0.5
                    ),
                    x = x_values,
                    y = y_values,
                    label = labels,
                    color = node_colors
                ),
                link = dict(
                    source = sources_2,
                    target = targets_2,
                    value = values_2,
                    color = link_colors_2
                ),
                domain={
                    'y': [0.6,0.98],
                    'x': [0,0.9] 
                }
            )
            legend = []
            legend_colors = legend_colors
            legend_names = legend_names
            for color,name in zip(legend_colors,legend_names):
                legend.append(
                    go.Scatter(
                        mode="markers",
                        x=[None],
                        y=[None],
                        marker=dict(size=10, color=color, symbol="square"),
                        name=name,
                    )
                )
            data = [trace1, trace2] + legend

            layout =  go.Layout(
                #title = dict(
                #    text = str(self.name[0])+' '+'2020 and 2030',
                #    xanchor = 'right',
                #    yanchor = 'bottom'
                #    ),
                showlegend = True,
                grid = None,
                
                font = dict(
                  size = 15
                )
            )

            fig_double = go.Figure(data=data, 
                            layout=layout)
            fig_double.update_xaxes(visible=False)
            fig_double.update_yaxes(visible=False)
            fig_double.update_layout(title_text=self.name[0]+' '+'2020 and 2030',font_size=20)
            #Set default renderer for graphic to browser
            pio.renderers.default = "browser"
            #Show figure with renderer
            #Save figure as website
            fig_double.show
            fig_double.write_html(filename_double),
            return fig_single.write_html(filename_single),fig_double.write_html(filename_double)
        else:
            print('No sankey diagram')
     
    
    def excelwriter(self,
                    filepath):
            # Run calculate_mfa_system
            flows_all,flows_to,d_stocks,stocks,rec_flows,pre_rec,post_rec,prod_virgin,df_import,production = self.calculate_mfa_system()
            
            # Create workbook
            workbook = pd.ExcelWriter(filepath)
            # Create sheets for each year 
            for t,i in zip(self.timesteps,range(len(flows_all))):
                flows_all[i]['Sum']=flows_all[i].sum(numeric_only=True,axis=1)
                flows_all[i].to_excel(workbook, sheet_name='flows_'+str(t))
            # Create sheet for delta stock and stock
            d_stocks.loc['Sum']=d_stocks.sum(numeric_only=True, axis=0)
            d_stocks.to_excel(workbook, sheet_name='d_stocks')
            stocks.loc['Sum']=stocks.sum(numeric_only=True, axis=0)
            stocks.to_excel(workbook, sheet_name='Stocks')
            # Create sheet for total recycling flows, pre-consumer, post_consumer, virgin and import flows
            rec_flows.loc['Sum']=rec_flows.sum(numeric_only=True, axis=0)
            rec_flows.to_excel(workbook, sheet_name='Total recycling flows input')
            post_rec.loc['Sum']=post_rec.sum(numeric_only=True, axis=0)
            post_rec.to_excel(workbook, sheet_name='Post consumer recycling flows')
            pre_rec.loc['Sum']=pre_rec.sum(numeric_only=True, axis=0)
            pre_rec.to_excel(workbook, sheet_name='Pre consumer recycling flows')
            prod_virgin.loc['Sum']=prod_virgin.sum(numeric_only=True, axis=0)
            prod_virgin.to_excel(workbook, sheet_name='Virgin production')
            #production.loc['Sum']=production.sum(numeric_only=True, axis=0)
            #production.to_excel(workbook, sheet_name='Total production')
            df_import.loc['Sum']=df_import.sum(numeric_only=True, axis=0)
            df_import.to_excel(workbook, sheet_name='Import flows')
            
            workbook.close()
            
def get_config():
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("The file 'config.yaml' was not found in the directory '" + os.getcwd() +"'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return config

config = get_config()
mfa = Model(config)
flows_all,flows_to,d_stocks,stocks,rec_flows,pre_rec,post_rec,prod_virgin,df_import,production = mfa.calculate_mfa_system()
mfa.excelwriter(config["output"]["filename"])
single = mfa.sankey_diagram(config["output"]["single_sankey_filename"],config["output"]["double_sankey_filename"])








