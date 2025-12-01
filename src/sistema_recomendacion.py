import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")

class SistemaRecomendacionApriori:
    def __init__(self):
        self.df = None
        self.rules = None
        self.frequent_itemsets = None
        self.transactions = None
        
    def generar_datos_ejemplo(self):
        """Generar dataset de ejemplo para análisis"""
        print("Generando dataset de ejemplo...")
        
        np.random.seed(42)
        
        productos = {
            'Electronics': [
                'Wireless Headphones', 'USB-C Charger', 'Phone Case',
                'Power Bank', 'Wireless Mouse', 'Tablet', 'Smartwatch'
            ],
            'Books': [
                'Python Programming', 'Machine Learning',
                'Data Science', 'Artificial Intelligence', 'Statistics'
            ],
            'Home': [
                'Coffee Maker', 'Desk Lamp', 'Knife Set',
                'Blender', 'Vacuum Cleaner', 'Toaster'
            ],
            'Sports': [
                'Running Shoes', 'Soccer Ball', 'Tennis Racket',
                'Dumbbells', 'Sportswear'
            ]
        }
        
        transactions = []
        users = [f'User_{i:03d}' for i in range(1, 201)]
        
        for user in users:
            n_purchases = np.random.randint(2, 8)
            user_categories = np.random.choice(list(productos.keys()), 2, replace=False)
            
            purchased_products = []
            for _ in range(n_purchases):
                category = np.random.choice(user_categories)
                product = np.random.choice(productos[category])
                purchased_products.append(product)
            
            unique_products = list(set(purchased_products))
            for product in unique_products:
                product_category = next(cat for cat, prods in productos.items() if product in prods)
                transactions.append({
                    'user_id': user,
                    'product': product,
                    'category': product_category,
                    'rating': np.random.uniform(3.5, 5.0),
                    'price': np.random.uniform(10, 150)
                })
        
        self.df = pd.DataFrame(transactions)
        print(f"Dataset creado: {len(self.df)} transacciones, {self.df['user_id'].nunique()} usuarios")
        return self.df
    
    def preparar_datos_apriori(self):
        """Preparar datos para algoritmo Apriori"""
        print("\nPreparando datos para Apriori...")
        
        self.transactions = self.df.groupby('user_id')['product'].apply(list).tolist()
        
        print(f"Transacciones: {len(self.transactions)}")
        print(f"Usuarios unicos: {self.df['user_id'].nunique()}")
        print(f"Productos unicos: {self.df['product'].nunique()}")
        
        return self.transactions
    
    def ejecutar_apriori(self, min_support=0.03, min_confidence=0.5):
        """Ejecutar algoritmo Apriori y generar reglas"""
        print(f"\nEjecutando algoritmo Apriori...")
        print(f"Parametros: min_support={min_support}, min_confidence={min_confidence}")
        
        # Codificar transacciones
        te = TransactionEncoder()
        te_array = te.fit(self.transactions).transform(self.transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)
        
        # Itemsets frecuentes
        self.frequent_itemsets = apriori(
            df_encoded, 
            min_support=min_support,
            use_colnames=True,
            max_len=3
        )
        
        print(f"Itemsets frecuentes encontrados: {len(self.frequent_itemsets)}")
        
        if len(self.frequent_itemsets) > 0:
            # Generar reglas de asociacion
            self.rules = association_rules(
                self.frequent_itemsets, 
                metric="confidence", 
                min_threshold=min_confidence
            )
            
            # Filtrar reglas utiles
            self.rules = self.rules[self.rules['lift'] > 1.0]
            self.rules = self.rules.sort_values(['confidence', 'lift'], ascending=[False, False])
            
            print(f"Reglas de asociacion generadas: {len(self.rules)}")
            
            # Mostrar metricas principales
            if len(self.rules) > 0:
                print(f"Confianza promedio: {self.rules['confidence'].mean():.3f}")
                print(f"Lift promedio: {self.rules['lift'].mean():.3f}")
                print(f"Soporte promedio: {self.rules['support'].mean():.3f}")
            
            return True
        else:
            print("No se encontraron itemsets frecuentes")
            return False
    
    def mostrar_reglas_top(self, n=10):
        """Mostrar las mejores reglas de asociacion"""
        if self.rules is None or len(self.rules) == 0:
            print("No hay reglas para mostrar")
            return
        
        print(f"\nTOP {n} REGLAS DE ASOCIACION:")
        print("=" * 70)
        
        for i, (idx, rule) in enumerate(self.rules.head(n).iterrows(), 1):
            antecedents = ", ".join(list(rule['antecedents']))
            consequents = ", ".join(list(rule['consequents']))
            
            print(f"{i}. SI: {antecedents}")
            print(f"   ENTONCES: {consequents}")
            print(f"   Soporte: {rule['support']:.4f} | Confianza: {rule['confidence']:.4f} | Lift: {rule['lift']:.4f}")
            print()
    
    def crear_visualizaciones_apriori(self):
        """Crear visualizaciones del analisis Apriori"""
        print("\nGenerando visualizaciones...")
        
        # Figura principal
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ANALISIS APRIORI - SISTEMA DE RECOMENDACION', fontsize=16, fontweight='bold')
        
        # 1. Productos mas populares
        self._grafica_productos_populares(axes[0, 0])
        
        # 2. Distribucion por categoria
        self._grafica_categorias(axes[0, 1])
        
        # 3. Reglas de asociacion
        self._grafica_reglas_asociacion(axes[0, 2])
        
        # 4. Itemsets frecuentes
        self._grafica_itemsets_frecuentes(axes[1, 0])
        
        # 5. Confianza vs Lift
        self._grafica_confianza_lift(axes[1, 1])
        
        # 6. Distribucion de soporte
        self._grafica_distribucion_soporte(axes[1, 2])
        
        plt.tight_layout()
        plt.show()
        
        # Visualizaciones adicionales
        self._visualizaciones_adicionales()
    
    def _grafica_productos_populares(self, ax):
        """Grafica de productos mas comprados"""
        top_products = self.df['product'].value_counts().head(10)
        
        bars = ax.barh(range(len(top_products)), top_products.values, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top_products)))
        ax.set_yticklabels(top_products.index)
        ax.set_xlabel('Numero de Compras')
        ax.set_title('Top 10 Productos Mas Populares', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    def _grafica_categorias(self, ax):
        """Grafica de distribucion por categorias"""
        category_dist = self.df['category'].value_counts()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        wedges, texts, autotexts = ax.pie(category_dist.values, labels=category_dist.index,
                                        autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Distribucion por Categorias', fontweight='bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _grafica_reglas_asociacion(self, ax):
        """Grafica de dispersion de reglas de asociacion"""
        if self.rules is not None and len(self.rules) > 0:
            scatter = ax.scatter(self.rules['support'], self.rules['confidence'],
                               c=self.rules['lift'], s=self.rules['lift']*80,
                               alpha=0.6, cmap='viridis')
            ax.set_xlabel('Soporte')
            ax.set_ylabel('Confianza')
            ax.set_title('Reglas de Asociacion (Size = Lift)', fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Lift')
            ax.grid(True, alpha=0.3)
    
    def _grafica_itemsets_frecuentes(self, ax):
        """Grafica de itemsets frecuentes"""
        if self.frequent_itemsets is not None and len(self.frequent_itemsets) > 0:
            top_itemsets = self.frequent_itemsets.nlargest(8, 'support')
            
            labels = []
            for items in top_itemsets['itemsets']:
                items_list = list(items)
                if len(items_list) == 1:
                    label = str(items_list[0])
                else:
                    label = " + ".join([str(item) for item in items_list])
                labels.append(label[:20] + '...' if len(label) > 20 else label)
            
            y_pos = np.arange(len(top_itemsets))
            bars = ax.barh(y_pos, top_itemsets['support'], color='lightgreen', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Soporte')
            ax.set_title('Itemsets Frecuentes', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _grafica_confianza_lift(self, ax):
        """Grafica de confianza vs lift"""
        if self.rules is not None and len(self.rules) > 0:
            scatter = ax.scatter(self.rules['confidence'], self.rules['lift'],
                               c=self.rules['support'], s=100, alpha=0.6, cmap='plasma')
            ax.set_xlabel('Confianza')
            ax.set_ylabel('Lift')
            ax.set_title('Confianza vs Lift', fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Soporte')
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Lift = 1')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _grafica_distribucion_soporte(self, ax):
        """Histograma de distribucion de soporte"""
        if self.frequent_itemsets is not None and len(self.frequent_itemsets) > 0:
            ax.hist(self.frequent_itemsets['support'], bins=15, alpha=0.7, 
                   color='purple', edgecolor='black')
            ax.set_xlabel('Soporte')
            ax.set_ylabel('Frecuencia')
            ax.set_title('Distribucion de Soporte', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    def _visualizaciones_adicionales(self):
        """Visualizaciones adicionales"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Boxplot de precios por categoria
        sns.boxplot(data=self.df, x='category', y='price', ax=ax1)
        ax1.set_title('Distribucion de Precios por Categoria', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylabel('Precio')
        
        # Heatmap de correlacion entre productos populares
        if self.rules is not None and len(self.rules) > 0:
            # Crear matriz de productos frecuentes
            top_products = list(self.frequent_itemsets.nlargest(10, 'support')['itemsets'])
            product_pairs = []
            
            for itemset in top_products:
                items_list = list(itemset)
                if len(items_list) >= 2:
                    for i in range(len(items_list)):
                        for j in range(i+1, len(items_list)):
                            product_pairs.append({
                                'product_a': items_list[i],
                                'product_b': items_list[j],
                                'support': self.frequent_itemsets[
                                    self.frequent_itemsets['itemsets'] == itemset
                                ]['support'].values[0]
                            })
            
            if product_pairs:
                df_pairs = pd.DataFrame(product_pairs)
                pivot_table = df_pairs.pivot_table(values='support', 
                                                 index='product_a', 
                                                 columns='product_b', 
                                                 fill_value=0)
                
                sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', ax=ax2, 
                           cbar_kws={'label': 'Soporte'})
                ax2.set_title('Relacion entre Productos Frecuentes', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def generar_recomendaciones(self, producto_input, n_recomendaciones=5):
        """Generar recomendaciones basadas en reglas de asociacion"""
        if self.rules is None or len(self.rules) == 0:
            print("No hay reglas de asociacion para generar recomendaciones")
            return []
        
        print(f"\nGenerando recomendaciones para: {producto_input}")
        print("-" * 50)
        
        recommendations = []
        
        for idx, rule in self.rules.iterrows():
            antecedents = list(rule['antecedents'])
            
            if producto_input in antecedents:
                for consequent in rule['consequents']:
                    if consequent != producto_input:
                        score = rule['confidence'] * rule['lift']
                        recommendations.append({
                            'producto': consequent,
                            'confianza': rule['confidence'],
                            'lift': rule['lift'],
                            'score': score
                        })
        
        if recommendations:
            df_rec = pd.DataFrame(recommendations)
            df_rec = df_rec.drop_duplicates('producto')
            df_rec = df_rec.sort_values('score', ascending=False)
            
            # Mostrar recomendaciones
            for i, (idx, rec) in enumerate(df_rec.head(n_recomendaciones).iterrows(), 1):
                print(f"{i}. {rec['producto']}")
                print(f"   Confianza: {rec['confianza']:.3f} | Lift: {rec['lift']:.3f} | Score: {rec['score']:.3f}")
            
            # Grafica de recomendaciones
            self._grafica_recomendaciones(df_rec.head(n_recomendaciones), producto_input)
            
            return df_rec.head(n_recomendaciones)
        else:
            print("No se encontraron recomendaciones para este producto")
            return []
    
    def _grafica_recomendaciones(self, recomendaciones, producto_input):
        """Grafica de barras para recomendaciones"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        productos = [rec['producto'] for _, rec in recomendaciones.iterrows()]
        scores = [rec['score'] for _, rec in recomendaciones.iterrows()]
        
        bars = ax.bar(productos, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_ylabel('Puntaje de Recomendacion')
        ax.set_title(f'Recomendaciones para: {producto_input}', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# Funcion de ejemplo para uso rapido
def ejecutar_analisis_completo():
    """Ejecutar analisis completo del sistema de recomendacion"""
    sistema = SistemaRecomendacionApriori()
    
    # 1. Generar datos
    sistema.generar_datos_ejemplo()
    
    # 2. Preparar datos para Apriori
    sistema.preparar_datos_apriori()
    
    # 3. Ejecutar Apriori
    sistema.ejecutar_apriori(min_support=0.03, min_confidence=0.4)
    
    # 4. Mostrar reglas
    sistema.mostrar_reglas_top(8)
    
    # 5. Visualizaciones
    sistema.crear_visualizaciones_apriori()
    
    return sistema