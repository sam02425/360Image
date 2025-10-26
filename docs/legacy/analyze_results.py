#!/usr/bin/env python3
"""
Analyze existing training results with proper statistical methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

class ExistingResultsAnalyzer:
    """Analyze existing experimental results with rigorous statistics"""
    
    def __init__(self, results_dir="results/training_runs"):
        self.results_dir = Path(results_dir)
        self.results_df = None
        self.statistical_results = {}
        
    def load_existing_results(self):
        """Load all existing training results"""
        print("Loading existing training results...")
        
        csv_files = glob.glob(str(self.results_dir / "*/results.csv"))
        print(f"Found {len(csv_files)} training result files")
        
        all_results = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                run_name = Path(csv_file).parent.name
                
                # Parse run information
                parts = run_name.split('_')
                if len(parts) >= 4:
                    architecture = parts[0]  # yolov8n or yolov11n
                    view_config = parts[1]   # dual, quad, etc.
                    run_info = parts[2]      # run0, run1, etc.
                    fold_info = parts[3]     # fold0
                    
                    run_idx = int(run_info.replace('run', ''))
                    fold_idx = int(fold_info.replace('fold', ''))
                    
                    # Get final metrics (last row)
                    if len(df) > 0:
                        final_row = df.iloc[-1]
                        
                        result = {
                            'experiment_id': run_name,
                            'architecture': architecture,
                            'view_config': view_config,
                            'run_idx': run_idx,
                            'fold_idx': fold_idx,
                            'map50': final_row.get('metrics/mAP50(B)', 0),
                            'map50_95': final_row.get('metrics/mAP50-95(B)', 0),
                            'precision': final_row.get('metrics/precision(B)', 0),
                            'recall': final_row.get('metrics/recall(B)', 0),
                            'f1_score': 0,  # Will calculate
                            'training_time_seconds': final_row.get('time', 0) * 60,  # Convert to seconds
                            'epochs_completed': len(df),
                            'converged': True,  # Assume converged if completed
                            'potential_overfitting': False,
                            'potential_data_leakage': False
                        }
                        
                        # Calculate F1 score
                        if result['precision'] + result['recall'] > 0:
                            result['f1_score'] = 2 * (result['precision'] * result['recall']) / (result['precision'] + result['recall'])
                        
                        all_results.append(result)
                        print(f"Loaded: {run_name} - mAP@0.5: {result['map50']:.3f}")
                        
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        if all_results:
            self.results_df = pd.DataFrame(all_results)
            print(f"\nLoaded {len(all_results)} experimental results")
            print("\nSummary by configuration:")
            summary = self.results_df.groupby(['architecture', 'view_config']).agg({
                'map50': ['count', 'mean', 'std'],
                'training_time_seconds': 'mean'
            }).round(3)
            print(summary)
            return True
        else:
            print("No valid results found")
            return False
    
    def calculate_descriptive_statistics(self):
        """Calculate comprehensive descriptive statistics"""
        if self.results_df is None or len(self.results_df) == 0:
            return {}
        
        metrics = ['map50', 'map50_95', 'precision', 'recall', 'f1_score']
        grouping_vars = ['architecture', 'view_config']
        
        stats_results = {}
        
        for group_name, group_df in self.results_df.groupby(grouping_vars):
            group_key = f"{group_name[0]}_{group_name[1]}"
            stats_results[group_key] = {}
            
            for metric in metrics:
                if metric in group_df.columns:
                    values = group_df[metric].dropna()
                    
                    if len(values) > 0:
                        stats_dict = {
                            'n': len(values),
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'median': float(values.median()),
                            'min': float(values.min()),
                            'max': float(values.max())
                        }
                        
                        # Confidence intervals
                        if len(values) > 1:
                            ci = stats.t.interval(
                                0.95, len(values) - 1,
                                loc=values.mean(),
                                scale=stats.sem(values)
                            )
                            stats_dict['ci_lower'] = float(ci[0])
                            stats_dict['ci_upper'] = float(ci[1])
                        
                        stats_results[group_key][metric] = stats_dict
        
        return stats_results
    
    def perform_statistical_tests(self):
        """Perform statistical significance tests"""
        if self.results_df is None or len(self.results_df) == 0:
            return {}
        
        metrics = ['map50', 'map50_95', 'precision', 'recall', 'f1_score']
        comparison_results = {}
        
        # Get unique groups
        groups = self.results_df.groupby(['architecture', 'view_config']).groups
        group_keys = list(groups.keys())
        
        for metric in metrics:
            metric_comparisons = {}
            p_values = []
            comparison_pairs = []
            
            # Pairwise comparisons
            for i, group1 in enumerate(group_keys):
                for j, group2 in enumerate(group_keys[i+1:], i+1):
                    
                    values1 = self.results_df[
                        (self.results_df['architecture'] == group1[0]) & 
                        (self.results_df['view_config'] == group1[1])
                    ][metric].dropna()
                    
                    values2 = self.results_df[
                        (self.results_df['architecture'] == group2[0]) & 
                        (self.results_df['view_config'] == group2[1])
                    ][metric].dropna()
                    
                    if len(values1) >= 2 and len(values2) >= 2:
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(values1, values2)
                        
                        # Calculate Cohen's d
                        pooled_std = np.sqrt(((len(values1) - 1) * values1.var() + 
                                             (len(values2) - 1) * values2.var()) / 
                                            (len(values1) + len(values2) - 2))
                        
                        if pooled_std > 0:
                            cohens_d = (values1.mean() - values2.mean()) / pooled_std
                        else:
                            cohens_d = 0
                        
                        comparison_key = f"{group1[0]}_{group1[1]}_vs_{group2[0]}_{group2[1]}"
                        
                        metric_comparisons[comparison_key] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'cohens_d': float(cohens_d),
                            'mean_diff': float(values1.mean() - values2.mean()),
                            'group1_mean': float(values1.mean()),
                            'group2_mean': float(values2.mean()),
                            'group1_n': len(values1),
                            'group2_n': len(values2),
                            'significant_uncorrected': p_value < 0.05
                        }
                        
                        p_values.append(p_value)
                        comparison_pairs.append(comparison_key)
            
            # Apply Bonferroni correction
            if p_values:
                rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                    p_values, method='bonferroni'
                )
                
                for i, pair in enumerate(comparison_pairs):
                    metric_comparisons[pair]['p_corrected'] = float(p_corrected[i])
                    metric_comparisons[pair]['significant_corrected'] = bool(rejected[i])
            
            comparison_results[metric] = metric_comparisons
        
        return comparison_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.results_df is None or len(self.results_df) == 0:
            print("No data available for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # Create main comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-View Object Detection Performance Analysis', fontsize=16, fontweight='bold')
        
        metrics = ['map50', 'map50_95', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//3, i%3]
            
            # Box plot
            sns.boxplot(data=self.results_df, x='view_config', y=metric, hue='architecture', ax=ax)
            sns.stripplot(data=self.results_df, x='view_config', y=metric, hue='architecture', 
                         ax=ax, dodge=True, alpha=0.7, size=4)
            
            ax.set_title(f'{metric.upper()} Comparison', fontweight='bold')
            ax.set_xlabel('View Configuration')
            ax.set_ylabel(f'{metric.upper()}')
            ax.grid(True, alpha=0.3)
            ax.legend(title='Architecture')
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig('experimental_results_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved visualization as 'experimental_results_analysis.png'")
        
        # Training time comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        sns.boxplot(data=self.results_df, x='view_config', y='training_time_seconds', hue='architecture', ax=ax)
        ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_xlabel('View Configuration')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved training time analysis as 'training_time_analysis.png'")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        if self.results_df is None:
            print("No data available for analysis")
            return
        
        # Calculate statistics
        descriptive_stats = self.calculate_descriptive_statistics()
        statistical_tests = self.perform_statistical_tests()
        
        # Generate report
        report_lines = [
            "# Multi-View Object Detection Experiment Analysis",
            "",
            "## Experimental Data Summary",
            f"- Total experiments completed: {len(self.results_df)}",
            f"- Architectures tested: {', '.join(self.results_df['architecture'].unique())}",
            f"- View configurations tested: {', '.join(self.results_df['view_config'].unique())}",
            "",
            "## Performance Results",
            ""
        ]
        
        # Add descriptive statistics
        for group, stats in descriptive_stats.items():
            report_lines.append(f"### {group.replace('_', ' ').title()}")
            report_lines.append("")
            report_lines.append("| Metric | Mean ± SD | 95% CI | N |")
            report_lines.append("|--------|-----------|--------|---|")
            
            for metric, metric_stats in stats.items():
                mean = metric_stats['mean']
                std = metric_stats['std']
                n = metric_stats['n']
                
                if 'ci_lower' in metric_stats:
                    ci_str = f"[{metric_stats['ci_lower']:.3f}, {metric_stats['ci_upper']:.3f}]"
                else:
                    ci_str = "N/A"
                
                report_lines.append(f"| {metric} | {mean:.3f} ± {std:.3f} | {ci_str} | {n} |")
            
            report_lines.append("")
        
        # Add statistical tests
        report_lines.append("## Statistical Significance Tests")
        report_lines.append("")
        
        for metric, comparisons in statistical_tests.items():
            if comparisons:
                report_lines.append(f"### {metric.upper()}")
                report_lines.append("")
                report_lines.append("| Comparison | Mean Diff | p-value | p-corrected | Cohen's d | Significant |")
                report_lines.append("|------------|-----------|---------|-------------|-----------|-------------|")
                
                for comp_name, comp_stats in comparisons.items():
                    comp_display = comp_name.replace('_vs_', ' vs ').replace('_', ' ')
                    mean_diff = comp_stats['mean_diff']
                    p_val = comp_stats['p_value']
                    p_corr = comp_stats.get('p_corrected', p_val)
                    cohens_d = comp_stats['cohens_d']
                    significant = "**Yes**" if comp_stats.get('significant_corrected', False) else "No"
                    
                    report_lines.append(f"| {comp_display} | {mean_diff:.3f} | {p_val:.4f} | {p_corr:.4f} | {cohens_d:.3f} | {significant} |")
                
                report_lines.append("")
        
        # Save report
        report_content = "\n".join(report_lines)
        with open('experimental_analysis_report.md', 'w') as f:
            f.write(report_content)
        
        print("Generated analysis report: experimental_analysis_report.md")
        
        # Also save results as JSON
        results_summary = {
            'descriptive_statistics': descriptive_stats,
            'statistical_tests': statistical_tests,
            'raw_data': self.results_df.to_dict('records')
        }
        
        with open('experimental_results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print("Saved detailed results: experimental_results_summary.json")
        
        return descriptive_stats, statistical_tests
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*60)
        print("RIGOROUS ANALYSIS OF EXISTING EXPERIMENTAL RESULTS")
        print("="*60)
        
        # Load results
        if not self.load_existing_results():
            print("No results found to analyze")
            return
        
        # Generate statistics and report
        descriptive_stats, statistical_tests = self.generate_report()
        
        # Create visualizations
        self.create_visualizations()
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Analyzed {len(self.results_df)} experimental runs")
        print("Files generated:")
        print("- experimental_analysis_report.md")
        print("- experimental_results_summary.json")
        print("- experimental_results_analysis.png")
        print("- training_time_analysis.png")
        
        # Print key findings
        if descriptive_stats:
            print("\nKey Findings:")
            for group, stats in descriptive_stats.items():
                if 'map50' in stats:
                    mean_map = stats['map50']['mean']
                    print(f"- {group}: mAP@0.5 = {mean_map:.3f}")

if __name__ == "__main__":
    analyzer = ExistingResultsAnalyzer()
    analyzer.run_complete_analysis()