from fractions import Fraction
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque,defaultdict
from math import lcm
from functools import reduce
import gradio as gr
import shutil, subprocess

dot = shutil.which("dot")
GraphWizInstalled=False

if dot:
    try:
        print(subprocess.check_output(["dot", "-V"], stderr=subprocess.STDOUT).decode().strip())
    except:
        print("other issues may be detected, no idea what though...")
    GraphWizInstalled=True
else:
    for i in range(10):
        print("GRAPHWIZ NOT FOUND - YOU LIKELY DID NOT INSTALL IT OR ITS NOT ON PATH, THE GRAPHS WILL LOOK BAD, PLEASE GET IT AT https://graphviz.org/download/")
#TODO orderining before simplfication - done
#TODO assure that each output has no duplicate inputs to avoid the sneaky split inefficecy
def GetFraction(Graph : nx.MultiDiGraph, Node, i=0): 
    if Node == "Output 0":
        #you can keep pulling form input, maybe? safer to just return this to prevent failure mode
        return (1,i)
    elif "Output" in str(Node):
        #you cannot pull from output without infinites
        return (0,i)
    Total=Fraction(0,1)
    for InEdge in Graph.in_edges(Node):
        Predecessor=InEdge[0]
        getFrac=GetFraction(Graph,Predecessor,i+1)
        i=max(i,getFrac[1])
        Total+=getFrac[0]
    #should i subtract the outedges? assume no for now
    Total*=Fraction(1,Graph.nodes[Node]['SplitterN'])
    return (Total,i) # i represents layer here

def layered_pos(
    G: nx.MultiDiGraph,
    roots=None,
    *,
    down_is_negative_y=True,
    y_gap=1.6,
    x_gap=1.8,
    component_gap=6.0,
):
    """
    Graphviz-free hierarchical-ish layout.

    - Builds a BFS spanning tree from one or more roots.
    - Assigns each node to a layer (depth).
    - Nodes on same layer get evenly spaced x positions.
    - Extra components (unreachable from roots) are laid out to the right.

    This is stable even with cycles (we don't "relax" layers via cycles).
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}

    # Pick roots if not provided
    if roots is None:
        if "Output 0" in G:
            roots = ["Output 0"]
        else:
            indeg0 = [n for n in nodes if G.in_degree(n) == 0]
            roots = indeg0 if indeg0 else [nodes[0]]

    # Ensure unique, existing
    roots = [r for r in dict.fromkeys(roots) if r in G]
    if not roots:
        roots = [nodes[0]]

    layer = {}
    order = []  # discovery order for stable sorting
    visited = set()

    def bfs_component(start, comp_index):
        q = deque([start])
        visited.add(start)
        if start not in layer:
            layer[start] = 0
        order.append(start)

        while q:
            u = q.popleft()
            # MultiDiGraph successors can repeat; dedupe
            for v in dict.fromkeys(G.successors(u)):
                if v not in visited:
                    visited.add(v)
                    layer[v] = layer[u] + 1
                    order.append(v)
                    q.append(v)

    # Layout main component(s) from roots first
    for r in roots:
        if r not in visited:
            layer[r] = 0
            bfs_component(r, 0)

    # Then any remaining components (place them to the right)
    comp = 0
    for n in nodes:
        if n not in visited:
            comp += 1
            layer[n] = 0
            bfs_component(n, comp)

    # Group nodes by (component, layer). We'll infer component by reachability “block” using x-offset.
    # Simple heuristic: nodes discovered later but with layer reset belong to later components.
    # We'll assign component ids by doing another BFS on undirected connectivity over unvisited? Too heavy.
    # Instead: compute weakly-connected components once (good enough).
    wcc = list(nx.weakly_connected_components(G))
    comp_id = {}
    for i, cc in enumerate(wcc):
        for n in cc:
            comp_id[n] = i

    by_layer = defaultdict(list)
    for n in nodes:
        by_layer[(comp_id.get(n, 0), layer.get(n, 0))].append(n)

    # stable ordering inside layer: discovery order, then str fallback
    order_index = {n: i for i, n in enumerate(order)}
    for k in by_layer:
        by_layer[k].sort(key=lambda n: (order_index.get(n, 10**9), str(n)))

    # Assign x/y
    pos = {}
    for (c, ly), ns in by_layer.items():
        m = len(ns)
        # center within layer
        for i, n in enumerate(ns):
            x = (i - (m - 1) / 2.0) * x_gap + c * component_gap
            y = (-ly if down_is_negative_y else ly) * y_gap
            pos[n] = (x, y)

    return pos
def draw_with_backedge_curves(
    G: nx.MultiDiGraph,
    down_is_negative_y=True,
    base_backedge_rad=0.25,
    arrowsize=20,
    node_size=500,
    rad_step=0.15
):
    """Draw directed graph with straight 'forward' edges and curved 'backward' edges, 
    spacing multiple edges between same nodes so they don't overlap."""
    #if I were to guess about 60% of this function is ai generated, its gui work though
    assert G.is_directed(), "Use a DiGraph/MultiDiGraph."
    if GraphWizInstalled:
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
    else:
        pos=layered_pos(G, roots=["Output 0"], down_is_negative_y=down_is_negative_y)
    def is_forward(u, v):
        yu, yv = pos[u][1], pos[v][1]
        if yv == yu:  # same level: treat as forward
            return True
        return (yv < yu) if down_is_negative_y else (yv > yu)

    # Count multiple edges
    edge_groups = defaultdict(list)
    GlobalRecursionSum=Fraction(0,1)
    for u, v, key in G.edges(keys=True):
        if v == "Output 0":
            GlobalRecursionSum+=GetFraction(G,u)[0]
        edge_groups[(u, v)].append(key)
    GlobalRecursionMultiplier=1/(1-GlobalRecursionSum)
    # Draw nodes and labels
    labels = {}

    for n in G.nodes():
        # maybe protect against missing attribute
        SplitterN=G.nodes[n].get('SplitterN', str(n))
        if SplitterN == 1:
            labels[n] = "INPUT"
        elif SplitterN == 0:
            sum=0
            for u, _ in G.in_edges(n):
                sum+=GetFraction(G,u)[0]
            labels[n]=sum*GlobalRecursionMultiplier
        else:
            labels[n] = SplitterN
    nx.draw_networkx_nodes(G, pos, node_size=node_size)
    nx.draw_networkx_labels(G, pos,labels=labels)

    for (u, v), keys in edge_groups.items():
        n = len(keys)
        if n == 1:
            # Single edge — normal logic
            if is_forward(u, v):
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], arrowstyle='->', arrowsize=arrowsize)
            else:
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)],
                    arrowstyle='->', arrowsize=arrowsize,
                    connectionstyle=f'arc3,rad={base_backedge_rad}'
                )
        else:
            # Multiple edges — spread them out
            radii = [base_backedge_rad + (i - (n - 1) / 2) * rad_step for i in range(n)]
            for i, rad in enumerate(radii):
                if is_forward(u, v):
                    nx.draw_networkx_edges(
                        G, pos, edgelist=[(u, v)],
                        arrowstyle='->', arrowsize=arrowsize,
                        connectionstyle=f'arc3,rad={rad}'
                    )
                else:
                    nx.draw_networkx_edges(
                        G, pos, edgelist=[(u, v)],
                        arrowstyle='->', arrowsize=arrowsize,
                        connectionstyle=f'arc3,rad={-rad}'
                    )

    plt.axis('off')
    plt.tight_layout()
    plt.show()
class BalancerReturn:
    def __init__(self):
        self.LCM=None
        self.TotalNumberOfOutputs=None
        self.OutputProportions=None
        self.NetworkSplitters=None

def FindBalancer(Pvalues : list, SplitterSet):
    SumOfFracs=sum(Pvalues)
    assert SumOfFracs <= 1
    if SumOfFracs < 1:
        Pvalues.append(1-SumOfFracs)
    splitterSolution=BalancerReturn()
    denominators=[]
    for Pvalue in Pvalues:
        denominators.append(Pvalue.denominator)
    LCM=reduce(lcm, denominators, 1)
    splitterSolution.LCM=LCM
    MinProduct=min_product_at_least_n_flat(LCM,SplitterSet)
    splitterSolution.TotalNumberOfOutputs=MinProduct[0]
    MinProduct=MinProduct[1:]
    splitterSolution.NetworkSplitters=MinProduct
    OutputDivisons=[splitterSolution.TotalNumberOfOutputs-splitterSolution.LCM]
    for Pvalue in Pvalues:
        OutputDivisons.append(LCM*Pvalue)
    splitterSolution.OutputProportions=OutputDivisons
    return splitterSolution
def GetAvailalbeNodes(G : nx.MultiDiGraph):
    for n in G.nodes:
        if 'SplitterN' not in G.nodes[n]:
            G.nodes[n]['SplitterN'] = 0
    return [n for n in G.nodes() if (len(list(G.successors(n))) < G.nodes[n]['SplitterN'])]
def GetAmountSpareOutputs(G : nx.MultiDiGraph, Node):
    return G.nodes[Node]['SplitterN']-G.out_degree(Node)

def IsGraphSimplified(G : nx.MultiDiGraph):
    for Node in G.nodes:
        Sucessors=list(G.out_edges(Node))
        if len(Sucessors) == 0 or len(Sucessors) == 1:
            continue
        if len(set([n[1] for n in G.out_edges(Node)])) == 1:
            #its just one item?
            return False
    return True
def SimplifyGraph(G : nx.MultiDiGraph):
    NodesToRemove=[]
    EdgesToAdd=[]
    for Node in G.nodes:
        Sucessors=list(G.out_edges(Node))
        if len(Sucessors) == 0 or len(Sucessors) == 1:
            continue
        StoreageToFigureIfSucessorsSame=Sucessors[0][1]
        BrokenFlag=False
        for output in G.out_edges(Node):
            if output[1] != StoreageToFigureIfSucessorsSame:
                BrokenFlag=True
                break
            StoreageToFigureIfSucessorsSame=output[1]
        if not BrokenFlag:
           # sucessor=EmptyGraph.successors(Node)[0] #tehcnially peormance gain possible here if i make it uhly
            Sucessor=Sucessors[0][1]
            for inbountconnection in G.in_edges(Node):
                EdgesToAdd.append((inbountconnection[0],Sucessor))
            NodesToRemove.append(Node)

    for Node in NodesToRemove:
        G.remove_node(Node)
    for Node in EdgesToAdd:
        G.add_edge(Node[0],Node[1])
    return G
def FullSimpliftyGraph(G : nx.MultiDiGraph):
    while not IsGraphSimplified(G):
        G=SimplifyGraph(G)
    return G

def min_product_at_least_n_flat(n: int, L: list[int]) -> list[int] | None:
    """
    Returns a flat list: [f1, f2, ..., fk, product]
    where product = f1*...*fk >= n, minimizing:
      (1) k (number of factors), then
      (2) product (closest above n).
    Repetitions allowed.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    factors = sorted({x for x in L if isinstance(x, int) and x > 1})
    if not factors:
        return None

    if n == 1:
        return [1]  # no factors; product is 1

    f_min, f_max = factors[0], factors[-1]

    # Upper bound on number of factors: multiply by largest until >= n
    p = 1
    k_ub = 0
    while p < n:
        p *= f_max
        k_ub += 1

    for k in range(1, k_ub + 1):
        best_prod = None
        best_list = None
        chosen = []

        def dfs(pos: int, start_idx: int, cur_prod: int):
            nonlocal best_prod, best_list
            remaining = k - pos

            # Can't reach n even using max for all remaining slots
            if cur_prod * (f_max ** remaining) < n:
                return

            # If already have a best, and even smallest completion >= best, prune
            if best_prod is not None and cur_prod * (f_min ** remaining) >= best_prod:
                return

            if pos == k:
                if cur_prod >= n and (best_prod is None or cur_prod < best_prod):
                    best_prod = cur_prod
                    best_list = chosen.copy()
                return

            for i in range(start_idx, len(factors)):
                f = factors[i]
                nxt = cur_prod * f

                # If we already have a best and even minimal completion >= best, break
                if best_prod is not None:
                    min_possible = nxt * (f_min ** (remaining - 1))
                    if min_possible >= best_prod:
                        break

                chosen.append(f)
                dfs(pos + 1, i, nxt)
                chosen.pop()

        dfs(0, 0, 1)

        if best_list is not None:
            return [best_prod]+best_list

    return None

def TurnSplitterSolutionIntoValidNetwork(SplitterSolution :BalancerReturn,globalnodeindex=0):
    EmptyGraph = nx.MultiDiGraph()
    EmptyGraph.add_node("Output 0", SplitterN=1)
    for K in range(1,len(SplitterSolution.OutputProportions)):
        EmptyGraph.add_node("Output "+str(K), SplitterN=0)
    #Convert The Splitter Vector into something in terms of the splitterset
    #Create A symmetirc Tree
    for Basis in SplitterSolution.NetworkSplitters: #check if splittervector actually is in the correct direciton
        for AvailableOutput in GetAvailalbeNodes(EmptyGraph):
            for vvv in range(GetAmountSpareOutputs(EmptyGraph,AvailableOutput)):
                EmptyGraph.add_node(globalnodeindex, SplitterN=Basis)
                EmptyGraph.add_edge(AvailableOutput,globalnodeindex)
                globalnodeindex+=1
    AvailableNodes=GetAvailalbeNodes(EmptyGraph)

    #nodes ordered based on proximity to eachover on the tree strcutture so that efficneicy is done
    i=0
    for Component in SplitterSolution.OutputProportions:
        for _ in range(int(Component)):
            if len(AvailableNodes) ==0:
                continue
            while GetAmountSpareOutputs(EmptyGraph,AvailableNodes[0]) == 0:
                AvailableNodes=AvailableNodes[1:]     
            try:
                EmptyGraph.add_edge(AvailableNodes[0],"Output " + str(i))
            except:
                print("Uh this should not happen")
            AvailableNodes=GetAvailalbeNodes(EmptyGraph)
        i+=1

    FullSimpliftyGraph(EmptyGraph)
   
    return EmptyGraph

def _parse_fraction_list(s: str):
    # accepts: "1/4, 2/7" or "0.25, 0.1"
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return [Fraction(p) for p in parts]

def _parse_int_list(s: str):
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return [int(p) for p in parts]

def _run(proportions_text: str, splitter_set_text: str):
    try:
        # Update global SplitterSet if user changes it
        SplitterSet = _parse_int_list(splitter_set_text)
        if not SplitterSet or any(x <= 1 for x in SplitterSet):
            raise ValueError("Splitter set must be comma-separated integers > 1 (e.g. 2,3,5).")
        SplitterSet.sort(reverse=True)

        Pvalues = _parse_fraction_list(proportions_text)
        if not Pvalues:
            raise ValueError("Provide at least one proportion (e.g. 1/4, 2/7).")
        # Run the solver
        sol = FindBalancer(Pvalues,SplitterSet)
        G = TurnSplitterSolutionIntoValidNetwork(sol)

        # --- capture your existing draw_with_backedge_curves() output as a Figure ---
        old_show = plt.show
        def _no_show(*args, **kwargs):

            return None
        plt.show = _no_show
        plt.close('all') #I actually added this line myself, gpt didint realize that everything stil saves to plt and isint cleared otherwise
        try:
            draw_with_backedge_curves(G)  # your function creates the plot
            fig = plt.gcf()
        finally:

            plt.show = old_show

        summary = (
            f"LCM: {sol.LCM}\n"
            f"Total Inital Outputs: {sol.TotalNumberOfOutputs}\n"
            f"Network splitters: {sol.NetworkSplitters}\n"
            f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}"
        )

        return summary, fig

    except Exception as e:
        fig = plt.figure(figsize=(6, 2))
        plt.axis("off")
        return f"Error: {e}", fig

with gr.Blocks(title="Splitter Network Builder") as demo:
    gr.Markdown("# Splitter Network Builder")
    if not GraphWizInstalled:
        gr.Markdown("# GRAPHWIZ NOT FOUND - YOU LIKELY DID NOT INSTALL IT OR ITS NOT ON PATH, THE GRAPHS WILL LOOK BAD, PLEASE GET IT AT https://graphviz.org/download/",)
    with gr.Row():
        proportions_in = gr.Textbox(
            label="Proportions (comma-separated Fractions) - tells the builder what outputs to aim for",
            value="1/4, 2/7, 13/28",
            placeholder="e.g. 1/4, 2/7, 13/28"
        )
        splitter_set_in = gr.Textbox(
            label="Splitter set (comma-separated ints > 1) - tells the builder what splitters its allowed to use",
            value="2, 3",
            placeholder="e.g. 2, 3"
        )


    run_btn = gr.Button("Build network", variant="primary")

    summary_out = gr.Textbox(label="Summary", lines=6)
    plot_out = gr.Plot(label="Graph")

    run_btn.click(_run, [proportions_in, splitter_set_in], [summary_out, plot_out])

if __name__ == "__main__":
    demo.launch()