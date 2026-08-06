import pandas as pd
import re

def parse_latex_table(tex_str, start_marker, end_marker):
    start = tex_str.find(start_marker)
    end = tex_str.find(end_marker, start)
    table_str = tex_str[start:end]
    lines = [line.strip() for line in table_str.split('\n') if line.strip() and not line.startswith('\\') and '&' in line]
    data = []
    for line in lines:
        row = [cell.strip() for cell in line.split('&')]
        row[-1] = row[-1].replace(r'\\', '').strip()
        data.append(row)
    return data

def parse_md_table(md_path):
    import io
    with open(md_path, 'r') as f:
        content = f.read()
    lines = [line for line in content.split('\n') if line.startswith('|')]
    # skip header and separator
    lines = lines[2:]
    data = []
    for line in lines:
        row = [cell.strip() for cell in line.split('|')[1:-1]]
        data.append(row)
    return data

def main():
    with open('report/Implementazione.tex', 'r') as f:
        tex = f.read()
        
    cat_tex = parse_latex_table(tex, r'\textbf{Dataset} & \textbf{Algorithm} & \textbf{SHD}', r'\bottomrule')
    sem_tex = parse_latex_table(tex, r'\textbf{Dataset} & \textbf{Algorithm} & \textbf{SHD}', r'\bottomrule') # wait, need to differentiate
    
    # Better way: find all tabulars
    tabulars = []
    for m in re.finditer(r'\\begin{tabular}{.*?}(.*?)\\end{tabular}', tex, re.DOTALL):
        lines = [line.strip() for line in m.group(1).split('\n') if line.strip() and not line.startswith('\\') and '&' in line]
        if not lines: continue
        data = []
        for line in lines:
            row = [cell.strip() for cell in line.split('&')]
            row[-1] = row[-1].replace(r'\\', '').strip()
            data.append(row)
        tabulars.append(data)
        
    cat_tex = tabulars[0]
    param_tex = tabulars[1]
    sem_tex = tabulars[2]
    
    md_data = parse_md_table('results/benchmarks/accuracy_report.md')
    md_cat = [r for r in md_data if r[0] == 'Categorical']
    md_sem = [r for r in md_data if r[0] == 'SEM']
    
    # Check categorical
    for t_row in cat_tex[1:]: # skip header
        ds, algo, shd, tp, fp, fn, f1, kl, hell, bhat, js = t_row
        ds = ds.replace(r'\_', '_')
        if algo == "FCI / PC / RFCI":
            algo_search = ["FCI_(Tetrad)", "PC_(Tetrad)", "RFCI_(Tetrad)", "FCI_Tetrad", "PC_Tetrad", "RFCI_Tetrad"]
        elif algo == "FCI / PC":
            algo_search = ["FCI_Tetrad", "PC_Tetrad"]
        else:
            algo_search = [algo.replace(' ', '_'), algo.replace(' (', '_(').replace(')','')]
            
        found = False
        for m_row in md_cat:
            m_ds = m_row[1]
            m_algo = m_row[2]
            
            if ds == m_ds and any(a in m_algo for a in algo_search):
                found = True
                assert int(shd) == int(m_row[3]), f"SHD mismatch: {shd} != {m_row[3]}"
                assert int(tp) == int(m_row[4]), f"TP mismatch"
                assert int(fp) == int(m_row[5]), f"FP mismatch"
                assert int(fn) == int(m_row[6]), f"FN mismatch"
                assert round(float(f1), 2) == round(float(m_row[7]), 2), f"F1 mismatch: {f1} != {m_row[7]}"
                assert round(float(kl), 3) == round(float(m_row[8]), 3), f"KL mismatch: {kl} != {m_row[8]}"
                assert round(float(hell), 3) == round(float(m_row[9]), 3), f"Hell mismatch"
                assert round(float(bhat), 3) == round(float(m_row[10]), 3), f"Bhat mismatch"
                break
        if not found:
            print(f"Could not find matching row for {ds} {algo}")
            
    # Check SEM
    for t_row in sem_tex[1:]:
        ds, algo, shd, tp, fp, fn, f1 = t_row
        ds = ds.replace(r'\_', '_')
        m_ds = ds + "_sem"
        found = False
        for m_row in md_sem:
            # md algo is like subset_500_std.csv_DAGMA or DAG_GNN
            clean_algo = algo.replace('-', '_')
            if m_ds == m_row[1] and clean_algo in m_row[2]:
                found = True
                assert int(shd) == int(m_row[3]), f"SHD mismatch"
                assert int(tp) == int(m_row[4]), f"TP mismatch"
                assert int(fp) == int(m_row[5]), f"FP mismatch"
                assert int(fn) == int(m_row[6]), f"FN mismatch"
                assert round(float(f1), 2) == round(float(m_row[7]), 2), f"F1 mismatch"
                break
        if not found:
            print(f"Could not find matching row for SEM {ds} {algo}")

    print("All tables match perfectly!")

if __name__ == "__main__":
    main()
