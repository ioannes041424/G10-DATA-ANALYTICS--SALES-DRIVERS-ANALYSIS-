import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from matplotlib.gridspec import GridSpec
from tabulate import tabulate
import warnings
from sklearn.exceptions import ConvergenceWarning
import io

class DataAnalyticsDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analytics")
        self.root.geometry("1800x780")

        self.file_path = None
        self.df = None

       
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")

        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()

       
        self.create_main_layout()

    def configure_styles(self):
        self.style.configure('Accent.TButton', 
                             background='#2962ff',
                             foreground='white',
                             padding=8,
                             font=('Helvetica', 9))
        self.style.configure('Title.TLabel',
                             font=('Helvetica', 28, 'bold'),
                             foreground='#2962ff',
                             padding=20)
        self.style.configure("Treeview", 
                             background="#ffffff",
                             fieldbackground="#ffffff", 
                             foreground="#333333")
        self.style.map('Treeview', background=[('selected', '#0078d7')])


    def create_main_layout(self):
       
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=1)

      
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))

       
        title = ttk.Label(header_frame, 
                          text="Sales Drivers Analysis Dashboard",
                          style='Title.TLabel')
        title.pack(side=tk.LEFT)

       
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT, padx=20)

        
        button_browse = ttk.Button(button_frame, 
                                   text="Load Dataset",
                                   style='Accent.TButton',
                                   command=self.browse_file)
        button_browse.pack(side=tk.LEFT, padx=5)

        button_describe = ttk.Button(button_frame,
                                     text="Analyze Data",
                                     style='Accent.TButton',
                                     command=self.perform_analysis)
        button_describe.pack(side=tk.LEFT, padx=5)

        
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

       
        self.create_data_view_tab()
        self.create_analysis_tab()
        self.create_visualization_tab()
        self.create_sales_analysis_tab()
        self.create_sales_tree_tab()

    def create_data_view_tab(self):
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text='Data View')

    def create_analysis_tab(self):
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text='Statistical Analysis')

       
        title_label = ttk.Label(
            self.analysis_frame,
            text="COMPREHENSIVE STATISTICAL ANALYSIS REPORT",
            font=('Helvetica', 15, 'bold'),
            foreground='#2962ff'
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(10, 15), sticky='nsew')
        self.analysis_frame.grid_columnconfigure(0, weight=1)



      
        self.stats_tree = ttk.Treeview(
            self.analysis_frame,
            show="headings",
            selectmode="none"
        )
        
        
        vsb = ttk.Scrollbar(self.analysis_frame, orient="vertical", command=self.stats_tree.yview)
        hsb = ttk.Scrollbar(self.analysis_frame, orient="horizontal", command=self.stats_tree.xview)
        self.stats_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
       
        self.stats_tree.grid(row=1, column=0, sticky='nsew')
        vsb.grid(row=1, column=1, sticky='ns')
        hsb.grid(row=2, column=0, sticky='ew')
        
        
        self.analysis_frame.grid_columnconfigure(0, weight=1)
        self.analysis_frame.grid_rowconfigure(1, weight=1)

    def create_visualization_tab(self):
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text='Data Visualization')

    def browse_file(self):
        try:
            self.file_path = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
            )
            if self.file_path:
               
                self.df = pd.read_csv(self.file_path) if self.file_path.endswith('.csv') \
                    else pd.read_excel(self.file_path)
                
               
                self.display_dataframe()
                messagebox.showinfo("Success", "Data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")

    def display_dataframe(self):
        
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        
        if self.df is not None:
            table_frame = ttk.Frame(self.data_frame)
            table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            tree = ttk.Treeview(table_frame, show="headings")
            
            tree["columns"] = list(self.df.columns)
            for col in self.df.columns:
                tree.heading(col, text=col)
                max_width = max(self.df[col].astype(str).map(len).max(), len(col)) * 10
                tree.column(col, width=min(max_width + 20, 300))
            
            for i, row in self.df.iterrows():
                tags = ('evenrow',) if i % 2 == 0 else ('oddrow',)
                tree.insert("", "end", values=list(row), tags=tags)
            
            tree.tag_configure('evenrow', background='#f5f5f5')
            tree.tag_configure('oddrow', background='#ffffff')
            
            vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            
            tree.grid(column=0, row=0, sticky='nsew')
            vsb.grid(column=1, row=0, sticky='ns')
            hsb.grid(column=0, row=1, sticky='ew')
            
            table_frame.grid_columnconfigure(0, weight=1)
            table_frame.grid_rowconfigure(0, weight=1)

    def perform_analysis(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return

        try:
           
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:
               
                self.display_statistical_summary(numeric_cols)
                
                
                self.create_data_visualizations(numeric_cols)
            else:
                messagebox.showinfo("Info", "No numeric columns found in the dataset.")

        except Exception as e:
            messagebox.showerror("Error", f"Error during analysis: {str(e)}")

    def display_statistical_summary(self, numeric_cols):
        
        headers = [
            "Column   ", 
            "Mean ", 
            "Range", 
            "Variance", 
            "Standard Deviation", 
            "Mean Deviation",
            "Quartile Deviation",
            "Coefficient of Range",
            "Coefficient of Variation", 
            "Coefficient of Mean Deviation",
            "Coefficient of Quartile Deviation"
        ]
        
        
        self.stats_tree["columns"] = headers
        
       
        for col in headers:
            self.stats_tree.heading(col, text=col, anchor=tk.CENTER)
            width = max(len(col) * 8, 150)  
            self.stats_tree.column(col, width=width, anchor=tk.CENTER)
        
        
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        
        for i, col in enumerate(numeric_cols):
            data = self.df[col]
            
           
            mean = data.mean()
            min_val = data.min()
            max_val = data.max()
            range_val = max_val - min_val
            variance = data.var()
            std_dev = data.std()
            mean_deviation = sum(abs(x - mean) for x in data) / len(data)
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            quartile_deviation = (q3 - q1) / 2
            
           
            coef_range = range_val / (max_val + min_val) 
            coef_variation = (std_dev / mean) * 100 if mean != 0 else 0
            coef_mean_dev = (mean_deviation / mean) * 100 if mean != 0 else 0
            coef_quartile_dev = (quartile_deviation / (q3+q1)) * 100 
            
        
            values = [
                col,
                f"{mean:.2f}",
                f"{range_val:.2f}",
                f"{variance:.2f}",
                f"{std_dev:.2f}",
                f"{mean_deviation:.2f}",
                f"{quartile_deviation:.2f}",
                f"{coef_range:.2f}",
                f"{coef_variation:.2f}%",
                f"{coef_mean_dev:.2f}%",
                f"{coef_quartile_dev:.2f}%"
            ]
            
            
            tags = ('evenrow',) if i % 2 == 0 else ('oddrow',)
            self.stats_tree.insert("", tk.END, values=values, tags=tags)
        
        
        self.stats_tree.tag_configure('evenrow', background='#f0f0f0')
        self.stats_tree.tag_configure('oddrow', background='#ffffff')

    def create_data_visualizations(self, numeric_cols):
       
        for widget in self.viz_frame.winfo_children():
            widget.destroy()

       

        
        fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig, wspace=0.2)
        
        
        ax1 = fig.add_subplot(gs[0, 0])
        
       
        price_bins = pd.qcut(self.df['Price'], q=8)
        avg_sales = self.df.groupby(price_bins)['Sales'].mean()
        
        
        bars = ax1.bar(range(len(avg_sales)), avg_sales.values,
                      color=sns.color_palette("husl", len(avg_sales)),
                      alpha=0.8,
                      width=0.5,  
                      edgecolor='black',
                      linewidth=1.0) 
        
        
        ax1.set_ylim(0, max(avg_sales.values) * 1.2)  
        
       
        for bar in bars:
            bar.set_zorder(1)
            bar_color = bar.get_facecolor()
            grad = np.linspace(0, 1, 2)
            gradient = np.vstack((grad, grad, grad))
            ax1.imshow(gradient.T, extent=[bar.get_x(), bar.get_x() + bar.get_width(),
                                         0, bar.get_height()], 
                      aspect="auto", zorder=0, alpha=0.4,
                      cmap=plt.cm.get_cmap('coolwarm'))
        
        
        ax1.set_title('Average Sales by Price Range', pad=15, fontsize=14, fontweight='bold')
        ax1.set_xlabel('Price Ranges', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Sales', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(avg_sales)))
        ax1.set_xticklabels([f'${int(i.left)}-${int(i.right)}' for i in avg_sales.index],
                           rotation=45, ha='right', fontsize=8)
       
        ax1.margins(x=0.1, y=0.1)
        
       
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}K',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
      
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax1.set_axisbelow(True)

        
        ax2 = fig.add_subplot(gs[0, 1])
        first_col = numeric_cols[0]
        bins = pd.cut(self.df[first_col], bins=5)
        bin_counts = bins.value_counts()
        
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return f'{pct:.1f}%\n({val:d})'
            return my_autopct

        colors = sns.color_palette("husl", n_colors=len(bin_counts))
        wedges, texts, autotexts = ax2.pie(bin_counts, 
                labels=[f'{int(x.left)}-{int(x.right)}' for x in bin_counts.index],
                autopct=make_autopct(bin_counts),
                startangle=90,
                colors=colors,
                explode=[0.05] * len(bin_counts),  
                shadow=True) 
        plt.setp(autotexts, size=9, weight="bold")
        plt.setp(texts, size=9)
        ax2.set_title(f'Distribution of {first_col} (Binned)', pad=20, fontsize=12, fontweight='bold')

        
        legend = ax2.legend(wedges, 
                       [f'{int(x.left)}-{int(x.right)}' for x in bin_counts.index],
                       title="Ranges",
                       loc="center left",
                       bbox_to_anchor=(1, 0, 0.5, 1))
    
        plt.tight_layout()  
        
        
        viz_container = ttk.Frame(self.viz_frame)
        viz_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = FigureCanvasTkAgg(fig, master=viz_container)
        canvas.draw()
        
       
        toolbar_frame = ttk.Frame(viz_container)
        toolbar_frame.pack(fill=tk.X, pady=(0, 5))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
       
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_sales_analysis_tab(self):
        self.sales_analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sales_analysis_frame, text='Sales Analysis')

       
        self.sales_analysis_text = scrolledtext.ScrolledText(
            self.sales_analysis_frame,
            wrap=tk.WORD,
            width=70,
            height=30,
            font=('Consolas', 10),
            background='#ffffff',
            foreground='#333333'
        )
        self.sales_analysis_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        
        control_frame = ttk.Frame(self.sales_analysis_frame)
        control_frame.pack(fill=tk.X, pady=10)

        
        sales_analysis_btn = ttk.Button(
            control_frame, 
            text="Perform Sales Analysis", 
            style='Accent.TButton',
            command=self.perform_sales_analysis
        )
        sales_analysis_btn.pack(side=tk.LEFT, padx=10)

    def create_sales_tree_tab(self):
        self.sales_tree_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sales_tree_frame, text='Decision Tree')

       
        self.tree_viz_container = ttk.Frame(self.sales_tree_frame)
        self.tree_viz_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

       
        tree_label = ttk.Label(self.tree_viz_container, 
                          text="Sales Decision Tree Visualization", 
                          font=('Helvetica', 14, 'bold'))
        tree_label.pack(fill=tk.X, pady=(0,10))

        
        self.tree_viz_frame = ttk.Frame(self.tree_viz_container)
        self.tree_viz_frame.pack(fill=tk.BOTH, expand=True)

        
        control_frame = ttk.Frame(self.sales_tree_frame)
        control_frame.pack(fill=tk.X, pady=10)

        analysis_btn = ttk.Button(
            control_frame, 
            text="Generate Decision Tree", 
            style='Accent.TButton',
            command=self.perform_sales_analysis
        )
        analysis_btn.pack(side=tk.LEFT, padx=10)
        

    def perform_sales_analysis(self):
        self.sales_analysis_text.config(state=tk.NORMAL)
        self.sales_analysis_text.delete(1.0, tk.END)
        

        try:
            file_path = filedialog.askopenfilename(
                title="Select Company Sales Data",
                filetypes=[("CSV files", "*.csv")]
            )
            
            if not file_path:
                messagebox.showinfo("Info", "No file selected")
                return
            
           
            df = pd.read_csv(file_path)
            
           
            required_columns = [
                'Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 
                'Price', 'ShelveLoc', 'Age', 'Education', 'Urban', 'US'
            ]
            
           
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                messagebox.showerror("Error", f"Missing columns: {', '.join(missing_columns)}")
                return

           
            sales_thresholds = [
                df['Sales'].quantile(0.5),
            ]
            df['Sales_Category'] = pd.cut(
                df['Sales'],
                bins=[-float('inf')] + sales_thresholds + [float('inf')],
                labels=['Low Sales', 'High Sales']
            )

           
            numerical_cols = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education']
            categorical_cols = ['ShelveLoc', 'Urban', 'US']
            
            
            X = df[numerical_cols + categorical_cols].copy()
            y = df['Sales_Category']

           
            categorical_mappings = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                categorical_mappings[col] = dict(zip(
                    le.classes_,
                    le.transform(le.classes_)
                ))

           
            param_grid = {
                'max_depth': [7, 8, 9], 
                'min_samples_split': [10, 15, 20],  
                'min_samples_leaf': [5, 7, 10],  
                'criterion': ['gini', 'entropy'],  
                'class_weight': ['balanced', None]
            }
            
            tree = DecisionTreeClassifier(random_state=42)
            grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X, y)
            
            best_tree = grid_search.best_estimator_

          
            cv_scores = cross_val_score(best_tree, X, y, cv=5)
            y_pred = best_tree.predict(X)
            accuracy = accuracy_score(y, y_pred)
            class_report = classification_report(y, y_pred)

           
            report = f"""
SALES DRIVERS ANALYSIS REPORT
============================

1. DATASET OVERVIEW
----------------
• Total Records: {len(df):,}
• Average Sales: ${df['Sales'].mean():.2f}K
• Sales Range: ${df['Sales'].min():.2f}K - ${df['Sales'].max():.2f}K
• Median Sales: ${df['Sales'].median():.2f}K
• Sales Std Dev: ${df['Sales'].std():.2f}K

2. MODEL PERFORMANCE METRICS
------------------------
• Overall Accuracy: {accuracy:.2%}
• Cross-validation Accuracy: {cv_scores.mean():.2%} (±{cv_scores.std()*2:.2%})



3. KEY SALES DRIVERS ANALYSIS
-------------------------
Top Influencing Factors (by importance):"""

           
           
           
          
           
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': best_tree.feature_importances_
            }).sort_values('Importance', ascending=False)

        
            for idx, row in feature_importance.iterrows():
                feature = row['Feature']
                importance = row['Importance']
                if importance >= 0.05: 
                    report += f"\n\n{feature} (Impact: {importance:.1%})"
                    if feature in numerical_cols:
                        median = df[feature].median()
                        high_sales = df[df[feature] > median]['Sales'].mean()
                        low_sales = df[df[feature] <= median]['Sales'].mean()
                        diff_pct = (high_sales - low_sales) / low_sales * 100
                        
                    
                        report += f"\n→ Median {feature}: {median:.2f}"
                        report += f"\n→ High vs Low {feature} difference: {np.abs(diff_pct):+.1f}% in sales"
                        
                      
                        
                    elif feature == 'ShelveLoc':
                    
                        for loc in ['Good', 'Medium', 'Bad']:
                            loc_data = df[df['ShelveLoc'] == loc]
                            avg_sales = loc_data['Sales'].mean()
                            sales_std = loc_data['Sales'].std()
                            count = len(loc_data)
                            report += f"\n→ {loc} shelf locations:"
                            report += f"\n  • Average sales: ${avg_sales:.2f}K (±${sales_std:.2f}K)"
                            report += f"\n  • Count: {count} ({count/len(df)*100:.1f}% of total)"
                            
                    elif feature in ['Urban', 'US']:
                    
                        for val in [0, 1]:
                            label = 'Yes' if val == 1 else 'No'
                            cat_data = df[df[feature] == val]
                            avg_sales = cat_data['Sales'].mean()
                            sales_std = cat_data['Sales'].std()
                            count = len(cat_data)
                            report += f"\n→ {feature}={label}:"
                            report += f"\n  • Average sales: ${avg_sales:.2f}K (±${sales_std:.2f}K)"
                            report += f"\n  • Count: {count} ({count/len(df)*100:.1f}% of total)"
                            ''' 
            
        report += "\n\n4. KEY RECOMMENDATIONS\n-------------------------"
            top_features = feature_importance.head(3)
            for _, row in top_features.iterrows():
                feature = row['Feature']
                if feature in numerical_cols:
                    optimal_value = df.loc[df['Sales'].idxmax(), feature]
                    report += f"\n• Optimize {feature}: Target value around {optimal_value:.2f} for maximum sales potential"
                elif feature == 'ShelveLoc':
                    best_loc = df.groupby('ShelveLoc')['Sales'].mean().idxmax()
                    report += f"\n• Prioritize {best_loc} shelf locations for optimal sales performance"
                elif feature in ['Urban', 'US']:
                    better_option = 'Yes' if df[df[feature] == 1]['Sales'].mean() > df[df[feature] == 0]['Sales'].mean() else 'No'
                    report += f"\n• Focus on {feature}={better_option} markets for better sales outcomes"
                    '''

            self.sales_analysis_text.insert(tk.END, report)
            self.sales_analysis_text.config(state=tk.DISABLED)

    
            plt.figure(figsize=(80, 30))
            plot_tree(best_tree, feature_names=X.columns, class_names=['High Sales', 'Low Sales'], filled=True, fontsize=6,label='none')
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(plt.gcf(), master=self.tree_viz_frame)
            canvas.draw()
            
            toolbar = NavigationToolbar2Tk(canvas, self.tree_viz_frame)
            toolbar.update()
            
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
           

         
            control_frame = ttk.Frame(self.sales_tree_frame)
            control_frame.pack(fill=tk.X, pady=10)


        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred: {str(e)}")
            self.sales_analysis_text.insert(tk.END, f"Error during analysis: {str(e)}")

def main():
    root = tk.Tk()
    app = DataAnalyticsDashboard(root)
    root.mainloop()
if __name__ == "__main__":
    main()