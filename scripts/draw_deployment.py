"""DecentralAI Deployment Topology Diagram"""
import sys, os
sys.path.insert(0, r'D:\pylib')
from PIL import Image, ImageDraw, ImageFont

W, H = 1400, 780
BG='#0d1117'; PANEL='#161b22'; BDR='#30363d'; TEXT='#e6edf3'
MUTED='#8b949e'; ACCENT='#6366f1'; GREEN='#22c55e'; BLUE='#3b82f6'
PURP='#a855f7'; YELL='#eab308'; CYAN='#06b6d4'; RED='#ef4444'

def _f(sz):
    for p in [r'C:\Windows\Fonts\segoeui.ttf',r'C:\Windows\Fonts\msyh.ttc',
              r'C:\Windows\Fonts\simhei.ttf',r'C:\Windows\Fonts\arial.ttf']:
        try: return ImageFont.truetype(p, sz)
        except: pass
    return ImageFont.load_default()

def rrect(d, xy, r=8, fill=None, outline=None, w=1):
    x1,y1,x2,y2 = xy
    if x1>x2: x1,x2=x2,x1
    if y1>y2: y1,y2=y2,y1
    r=min(r,(x2-x1)//2,(y2-y1)//2)
    if fill:
        d.rectangle([x1+r,y1,x2-r,y2],fill=fill)
        d.rectangle([x1,y1+r,x2,y2-r],fill=fill)
        d.ellipse([x1,y1,x1+2*r,y1+2*r],fill=fill)
        d.ellipse([x2-2*r,y1,x2,y1+2*r],fill=fill)
        d.ellipse([x1,y2-2*r,x1+2*r,y2],fill=fill)
        d.ellipse([x2-2*r,y2-2*r,x2,y2],fill=fill)
    if outline:
        d.line([x1+r,y1,x2-r,y1],fill=outline,width=w)
        d.line([x1,y1+r,x1,y2-r],fill=outline,width=w)
        d.line([x1+r,y2,x2-r,y2],fill=outline,width=w)
        d.line([x2,y1+r,x2,y2-r],fill=outline,width=w)
        d.arc([x1,y1,x1+2*r,y1+2*r],180,270,fill=outline,width=w)
        d.arc([x2-2*r,y1,x2,y1+2*r],270,360,fill=outline,width=w)
        d.arc([x1,y2-2*r,x1+2*r,y2],90,180,fill=outline,width=w)
        d.arc([x2-2*r,y2-2*r,x2,y2],0,90,fill=outline,width=w)

def label(d, cx, cy, txt, color, fsz):
    d.text((cx,cy), txt, fill=color, font=_f(fsz), anchor='mm')

def main():
    img = Image.new('RGB',(W,H), BG)
    d = ImageDraw.Draw(img)

    # Title
    label(d, W//2, 28, 'DecentralAI Deployment Topology', TEXT, 22)
    label(d, W//2, 56, 'From single node to multi-node cluster to on-chain governance', MUTED, 10)

    # ====== LEFT: Single Node ======
    label(d, 220, 90, 'Single Node', TEXT, 14)
    rrect(d, [35, 108, 405, H-15], 12, PANEL, BDR)

    items = [
        ('OS',       'Ubuntu 22.04 / Win / RPi OS', BLUE),
        ('Python',   '3.12 + pip + venv',            GREEN),
        ('Core',     'run.py / api_server.py / ws',  ACCENT),
        ('Model',    'RWKV-169M (CPU) / Qwen (GPU)', PURP),
        ('P2P',      'WebSocket :8001',              CYAN),
        ('Dashboard','http://localhost:8000/',        YELL),
        ('Chain',    'Optional: Scroll / zkSync',     RED),
    ]
    for i,(t,s,c) in enumerate(items):
        y = 124 + i*68
        rrect(d, [50, y, 390, y+58], 6, BG, c)
        label(d, 220, y+20, t, c, 11)
        label(d, 220, y+42, s, MUTED, 9)

    # Quick start box
    yb = 124 + 7*68 + 8
    rrect(d, [50, yb, 390, H-28], 6, '#0d1117', GREEN)
    label(d, 220, yb+14, 'Quick Start', GREEN, 11)
    cmds = ['pip install websockets pyyaml',
            'cp config.example.yaml config.yaml',
            'python run.py --level L1',
            'python api_server.py']
    for i,c in enumerate(cmds):
        label(d, 220, yb+34+i*16, f'$ {c}', GREEN, 8)

    # ====== Arrow ======
    ax = 418
    ay = H//2
    d.line([ax,ay,ax+45,ay], fill=MUTED, width=2)
    d.polygon([(ax+45,ay),(ax+38,ay-5),(ax+38,ay+5)], fill=MUTED)
    label(d, ax+22, ay+18, 'or', MUTED, 9)

    # ====== RIGHT: Multi-Node ======
    rx = 475
    label(d, rx + (W-rx)//2, 90, 'Multi-Node Cluster', TEXT, 14)
    rrect(d, [rx, 108, W-15, H-15], 12, BG, BDR)
    label(d, rx + (W-rx)//2, 125, 'Internet / LAN', MUTED, 10)

    # Node cards 3x2
    nodes = [
        ('node-1','L2',GREEN,'7B Q4 / GPU 3060'),
        ('node-2','L1',BLUE,'1.5B / CPU only'),
        ('node-3','L3',PURP,'14B+ / 3090/4090'),
        ('node-4','L0',MUTED,'Collector / CPU'),
        ('node-5','L1',YELL,'0.5B / CPU / ¥15/mo'),
        ('node-6','L4',ACCENT,'A100 / Cloud'),
    ]
    NW, NH = 165, 105
    cols, mx, my = 3, 20, 14
    row_h = NH + my
    col_w = NW + mx
    gx = rx + (W-15-rx - cols*col_w + mx)//2

    centers = []
    for i,(nm,nl,nc,ns) in enumerate(nodes):
        col, row = i%cols, i//cols
        nx2 = gx + col*col_w
        ny2 = 145 + row*row_h
        rrect(d, [nx2, ny2, nx2+NW, ny2+NH], 8, PANEL, nc)
        label(d, nx2+NW//2, ny2+18, nm, TEXT, 10)
        bw2, bh2 = 56, 28
        rrect(d, [nx2+NW//2-bw2//2, ny2+28, nx2+NW//2+bw2//2, ny2+28+bh2], 4, nc)
        label(d, nx2+NW//2, ny2+28+bh2//2, nl, 'white', 14)
        label(d, nx2+NW//2, ny2+75, ns, MUTED, 8)
        centers.append((nx2+NW//2, ny2+NH//2))

    # P2P lines
    pairs = [(0,1),(1,2),(0,3),(1,4),(2,5),(3,4),(4,5)]
    for a,b in pairs:
        x1,y1 = centers[a]; x2,y2 = centers[b]
        d.line([x1,y1,x2,y2], fill=BDR, width=1)

    # Blockchain layer
    by = 145 + 2*row_h + 25
    rrect(d, [rx+15, by, W-30, H-30], 8, '#1a1520', YELL)
    label(d, rx+(W-15-rx)//2, by+16, 'Blockchain Layer (Optional)', YELL, 11)

    chains = [
        ('Scroll','L2 recommended ~$0.01/tx',ACCENT),
        ('zkSync','L2 zkEVM low fee',PURP),
        ('长安链','Consortium CN compliance',GREEN),
        ('Arbitrum','L2 largest ecosystem',BLUE),
        ('Polygon','L2 mature ecosystem','#8247e5'),
        ('Hardhat','Local dev testnet',YELL),
    ]
    for i,(ct,cs,cc) in enumerate(chains):
        col2, row2 = i%3, i//3
        cx2 = rx+35 + col2*195
        cy2 = by+35 + row2*42
        rrect(d, [cx2, cy2, cx2+175, cy2+34], 4, BG, cc)
        label(d, cx2+87, cy2+12, ct, cc, 9)
        label(d, cx2+87, cy2+26, cs, MUTED, 7)

    # Dotted lines from top row to blockchain
    for ci in range(3):
        cx3 = centers[ci][0]
        for yy in range(int(centers[ci][1]+NH//2), int(by), 6):
            if yy%12<6:
                d.point((cx3,yy), fill=YELL)

    # Footer
    label(d, W//2, H-10, 'DecentralAI v0.3.0 | MIT License', MUTED, 8)

    out = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'deployment.png')
    out = os.path.abspath(out)
    img.save(out, 'PNG', optimize=True)
    print(f'Saved: {out}')

if __name__ == '__main__':
    main()
