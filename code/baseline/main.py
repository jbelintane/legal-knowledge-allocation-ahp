from pathlib import Path
# %% 1. Imports
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# %% 3. Carregar dados
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

df = pd.read_excel(DATA_DIR / "KPMG_Legal_Dept_tab.xlsx", header=0)
DB_AT = pd.read_excel(DATA_DIR / "DB_AT.xlsx", header=1)
DB_AN = pd.read_excel(DATA_DIR / "DB_AN.xlsx", header=1)

# %% 4. Limpeza básica
DB_AT = DB_AT.drop(columns=["Unnamed: 0"], errors="ignore")
DB_AN = DB_AN.drop(columns=["Unnamed: 0"], errors="ignore")

# %% 5. Mapeamento de áreas jurídicas
mapa = {
    "Contract Law": "Contratos",
    "Corporate Law": "Societário",
    "Antitrust Law": "Concorrencial",
    "Compliance Law": "Governança",
    "Litigation and Arbitration": "Contencioso",
    "M&A": "Societário",
    "General Terms and Conditions": "Contratos",
    "Data protection": "Direito digital",
    "International Legal affairs": "Contratos",
    "Trademark and patent law": "Propriedade intelectual",
    "Labor Law": "Trabalhista",
    "Banking law": "Direito financeiro",
    "IT Law": "Direito digital",
    "Financing": "Direito financeiro",
    "Energy industry law": "Regulatório",
    "Capital market law": "Regulatório",
    "Supervisory board office/executive board support": "Governança",
    "Product liability": "Contratos",
    "Public law": "Regulatório",
    "Property and building law": "Contratos",
    "insurance law": "Direito financeiro",
    "Environmental law": "Regulatório",
    "Stock corporation law and group law": "Societário",
    "Foreign trade/export controls": "Contratos",
    "Transport and logistics law": "Regulatório",
    "Insolvency law": "Societário",
    "Criminal law": "Contencioso",
    "Media law": "Regulatório",
    "Telecommunications": "Regulatório",
    "Tax law": "Tributário",
    "Press law": "Regulatório"
}

# %% 6. Criar coluna de agrupamento e distribuição de grupos
df["Grupo"] = df["Área"].map(mapa)

df_grupos = (
    df.groupby("Grupo", as_index=False)["Distribuição"]
      .sum()
      .copy()
)

df_grupos["p"] = df_grupos["Distribuição"] / df_grupos["Distribuição"].sum()

# %% 7. Validar importação / consistência entre bases
grupos_kpmg = set(df_grupos["Grupo"].dropna().unique())
areas_at = set(DB_AT["area"].dropna().unique())
areas_an = set(DB_AN["areaprin_dban"].dropna().unique())

print("Grupos KPMG:", sorted(grupos_kpmg))
print("Áreas DB_AT:", sorted(areas_at))
print("Áreas DB_AN:", sorted(areas_an))

print("\nGrupos da KPMG sem correspondência no DB_AT:")
print(sorted(grupos_kpmg - areas_at))

print("\nGrupos da KPMG sem correspondência no DB_AN:")
print(sorted(grupos_kpmg - areas_an))

print("\nÁreas do DB_AT que não estão na KPMG:")
print(sorted(areas_at - grupos_kpmg))

print("\nÁreas do DB_AN que não estão na KPMG:")
print(sorted(areas_an - grupos_kpmg))

# %% 8. Estrutura do DB_WK
colunas_db_wk = [
    "area_wk",
    "tarefa_WK",
    "descritivo",
    "analista",
    "data_aloc",
    "inicio_wk",
    "fim_wk",
    "comp_est",
    "comp_real",
    "tempo_est_h",
    "prazo_est",
    "prazo_real",
    "crit_aloc",
    "resum_exec",
    "data_envio",
    "data_aval",
    "retorno",
    "status",
    "xp_creditada",   # nova coluna: controla se a tarefa já gerou experiência
]

# %% 9. Preparos da simulação
rng = np.random.default_rng(123)

data_inicio = pd.Timestamp("2026-01-01")
tarefas_por_dia = 5
total_tarefas = 1000
total_dias_uteis = math.ceil(total_tarefas / tarefas_por_dia)

grupos = df_grupos["Grupo"].astype(str).to_list()
probs = df_grupos["p"].astype(float).to_list()

# =========================================================
# 10) Funções auxiliares de calendário útil
# =========================================================
HORA_INICIO_EXP = 9
HORA_FIM_EXP = 17

def eh_dia_util(ts: pd.Timestamp) -> bool:
    return pd.Timestamp(ts).weekday() < 5  # 0=segunda, 6=domingo

def inicio_expediente(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.normalize() + pd.Timedelta(hours=HORA_INICIO_EXP)

def fim_expediente(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.normalize() + pd.Timedelta(hours=HORA_FIM_EXP)

def ajustar_para_horario_util(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Ajusta um timestamp para o próximo instante útil:
    - se cair em fim de semana, vai para a próxima segunda às 09:00;
    - se cair antes das 09:00, ajusta para 09:00 do mesmo dia útil;
    - se cair às 17:00 ou depois, vai para o próximo dia útil às 09:00;
    - se já estiver dentro do expediente útil, retorna o próprio instante.
    """
    atual = pd.Timestamp(ts)

    while True:
        if not eh_dia_util(atual):
            atual = atual.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=HORA_INICIO_EXP)
            continue

        ini = inicio_expediente(atual)
        fim = fim_expediente(atual)

        if atual < ini:
            return ini

        if atual >= fim:
            atual = atual.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=HORA_INICIO_EXP)
            continue

        return atual

def gerar_dias_uteis(inicio: pd.Timestamp, quantidade: int):
    """
    Gera uma lista de datas-base úteis (normalizadas) a partir de 'inicio'.
    """
    dias = []
    atual = pd.Timestamp(inicio).normalize()

    while len(dias) < quantidade:
        if eh_dia_util(atual):
            dias.append(atual)
        atual = atual + pd.Timedelta(days=1)

    return dias

def timestamp_do_dia_util(dia_base: pd.Timestamp, idx_no_dia: int) -> pd.Timestamp:
    """
    Gera timestamps de alocação sequenciais dentro do dia útil:
    09:00, 09:01, 09:02, ...
    """
    return inicio_expediente(dia_base) + pd.Timedelta(minutes=idx_no_dia)

def horas_uteis_entre(inicio: pd.Timestamp, fim: pd.Timestamp) -> float:
    """
    Calcula horas úteis entre dois timestamps, considerando:
    - segunda a sexta
    - 09:00 às 17:00
    - sem considerar feriados
    """
    if pd.isna(inicio) or pd.isna(fim):
        return 0.0

    inicio = pd.Timestamp(inicio)
    fim = pd.Timestamp(fim)

    if fim <= inicio:
        return 0.0

    atual = inicio
    horas = 0.0

    while atual < fim:
        atual = ajustar_para_horario_util(atual)
        if atual >= fim:
            break

        fim_dia = fim_expediente(atual)
        limite = min(fim, fim_dia)

        if limite > atual:
            horas += (limite - atual).total_seconds() / 3600.0

        # vai para o próximo dia útil
        atual = atual.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=HORA_INICIO_EXP)

    return float(horas)

def calcular_fim_estimado(inicio: pd.Timestamp, horas_necessarias: float) -> pd.Timestamp:
    """
    Avança no calendário útil até consumir 'horas_necessarias',
    respeitando jornada de 09:00 às 17:00 e pulando fins de semana.
    """
    atual = ajustar_para_horario_util(inicio)
    horas_restantes = float(horas_necessarias)

    while horas_restantes > 1e-9:
        atual = ajustar_para_horario_util(atual)
        fim_dia = fim_expediente(atual)
        horas_disponiveis = (fim_dia - atual).total_seconds() / 3600.0

        if horas_restantes <= horas_disponiveis + 1e-9:
            return atual + pd.Timedelta(hours=horas_restantes)

        horas_restantes -= horas_disponiveis
        atual = atual.normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=HORA_INICIO_EXP)

    return atual

#%% 

# =========================================================
# 11) Funções de negócio
# =========================================================
def sortear_grupo(rng: np.random.Generator) -> str:
    return rng.choice(grupos, p=probs)

def preparar_tarefas_por_area(DB_AT: pd.DataFrame) -> dict:
    """
    Pré-organiza as tarefas por área para evitar filtrar DB_AT
    a cada sorteio dentro da simulação.
    """
    tarefas_area = {}

    for area, sub in DB_AT.groupby("area"):
        tarefas_area[str(area)] = sub.reset_index(drop=True).copy()

    return tarefas_area


def sortear_tarefa_no_grupo_rapido(tarefas_area: dict,
                                   grupo: str,
                                   rng: np.random.Generator):
    """
    Sorteia uma tarefa já a partir do dicionário pré-filtrado por área.
    """
    fat = tarefas_area.get(str(grupo))

    if fat is None or fat.empty:
        raise ValueError(
            f"Não há tarefas no DB_AT para grupo='{grupo}'. "
            f"Verifique a padronização entre df_grupos['Grupo'] e DB_AT['area']."
        )

    idx = int(rng.integers(0, len(fat)))
    linha = fat.iloc[idx]

    tarefa = str(linha["Tarefa"])
    comp_est = float(linha["compl_dbat"])
    tempo_est_h = float(linha["horas_dbat"])

    return tarefa, comp_est, tempo_est_h

def horas_remanescentes_tarefa(row: pd.Series, agora: pd.Timestamp) -> float:
    """
    Calcula as horas remanescentes reais de uma tarefa no instante 'agora'.
    Regras:
    - se ainda não começou: remanescente = tempo_est_h
    - se está em execução: remanescente = horas úteis entre agora e fim_wk
    - se já terminou: remanescente = 0
    """
    inicio = row["inicio_wk"]
    fim = row["fim_wk"]
    tempo = float(row["tempo_est_h"])

    if pd.isna(inicio) or pd.isna(fim):
        return 0.0

    agora = pd.Timestamp(agora)

    if agora < inicio:
        return tempo

    if inicio <= agora < fim:
        return horas_uteis_entre(agora, fim)

    return 0.0

def workload_remanescente_analista(DB_WK: pd.DataFrame, analista: str, agora: pd.Timestamp) -> float:
    """
    Soma as horas remanescentes reais de todas as tarefas do analista
    no instante 'agora'.
    """
    if DB_WK.empty:
        return 0.0

    tarefas = DB_WK[DB_WK["analista"] == analista].copy()
    if tarefas.empty:
        return 0.0

    carga = tarefas.apply(lambda row: horas_remanescentes_tarefa(row, agora), axis=1).sum()
    return float(carga)

def proxima_disponibilidade_analista(DB_WK: pd.DataFrame, analista: str, agora: pd.Timestamp) -> pd.Timestamp:
    """
    Retorna o próximo instante em que o analista estará disponível.
    Como o modelo é sequencial, isso corresponde ao maior fim_wk das tarefas
    ainda existentes na fila histórica do analista, comparado com 'agora'.
    """
    agora = ajustar_para_horario_util(agora)

    if DB_WK.empty:
        return agora

    tarefas = DB_WK[DB_WK["analista"] == analista].copy()
    if tarefas.empty:
        return agora

    ultimo_fim = tarefas["fim_wk"].max()

    if pd.isna(ultimo_fim):
        return agora

    return ajustar_para_horario_util(max(agora, ultimo_fim))

def escolher_analista_AHP(DB_AN: pd.DataFrame, DB_WK: pd.DataFrame,
                          grupo: str, tarefa: str, comp_est: float,
                          tempo_est_h: float, agora: pd.Timestamp) -> str:
    """
    Regra atual de alocação:
    1) prioriza especialistas da área da tarefa (areaprin_dban == grupo);
    2) dentro dos elegíveis, escolhe o de menor workload remanescente real em horas;
    3) se não houver especialista na área, escolhe entre todos.
    """
    if "nome" not in DB_AN.columns:
        raise ValueError("DB_AN não tem coluna 'nome'.")
    if "areaprin_dban" not in DB_AN.columns:
        raise ValueError("DB_AN não tem coluna 'areaprin_dban'.")

    analistas_df = DB_AN[["nome", "areaprin_dban"]].copy()
    analistas_df["nome"] = analistas_df["nome"].astype(str)
    analistas_df["areaprin_dban"] = analistas_df["areaprin_dban"].astype(str)

    if analistas_df.empty:
        raise ValueError("DB_AN está vazio.")

    analistas_df["workload_h"] = analistas_df["nome"].apply(
        lambda nome: workload_remanescente_analista(DB_WK, nome, agora)
    )

    especialistas = analistas_df[analistas_df["areaprin_dban"] == grupo].copy()
    elegiveis = especialistas if not especialistas.empty else analistas_df

    elegiveis = elegiveis.sort_values(
        by=["workload_h", "nome"],
        ascending=[True, True],
        kind="stable"
    )

    return elegiveis.iloc[0]["nome"]

def classificar_status(inicio: pd.Timestamp, fim: pd.Timestamp, agora_ref: pd.Timestamp) -> str:
    """
    Classifica o status da tarefa em um dado instante de referência.
    """
    if pd.isna(inicio) or pd.isna(fim):
        return "INDEFINIDO"

    agora_ref = pd.Timestamp(agora_ref)

    if agora_ref < inicio:
        return "PENDENTE"

    if inicio <= agora_ref < fim:
        return "EM_EXECUCAO"

    return "CONCLUIDA"

def escolher_analista_rapido(estado_analistas: pd.DataFrame,
                             grupo: str) -> str:
    """
    Escolhe o analista com base em:
    1) especialização na área
    2) menor instante de disponibilidade

    Esta versão substitui o recálculo integral de workload histórico.
    """
    base = estado_analistas.copy()

    especialistas = base[base["area_principal"] == str(grupo)].copy()
    elegiveis = especialistas if not especialistas.empty else base

    elegiveis = elegiveis.sort_values(
        by=["disponivel_em", "analista"],
        ascending=[True, True],
        kind="stable"
    )

    return elegiveis.iloc[0]["analista"]


def inicializar_estado_analistas(DB_AN_cenario: pd.DataFrame,
                                 data_inicio: pd.Timestamp) -> pd.DataFrame:
    """
    Cria a base incremental de estado dos analistas.
    """
    instante_inicial = ajustar_para_horario_util(pd.Timestamp(data_inicio))

    estado = DB_AN_cenario[["nome", "areaprin_dban"]].copy()
    estado["nome"] = estado["nome"].astype(str)
    estado["areaprin_dban"] = estado["areaprin_dban"].astype(str)

    estado = estado.rename(columns={
        "nome": "analista",
        "areaprin_dban": "area_principal"
    })

    estado["disponivel_em"] = instante_inicial
    return estado

#%% 

# =========================================================
# 11-A) Funções de aprendizagem / conhecimento
# =========================================================

# Parâmetros da curva sigmoide
# b = inclinação da curva
# m = ponto médio da curva
B_SIGMOIDE = 0.0706
M_SIGMOIDE = 24.0


def calcular_xp_tarefa(comp_est: float, tempo_est_h: float) -> float:
    """
    Calcula a unidade de experiência prática da tarefa.
    Regra atual:
        xp = complexidade estimada × tempo estimado em horas x fator de escala
        
        O fator de escala reduz a velocidade de crescimento da experiência,
        evitando saturação precoce da curva sigmoide.
        
        
    """
    fator_escala_xp = 0.05
    return float(comp_est) * float(tempo_est_h) * fator_escala_xp


def calcular_conhecimento_sigmoide(xp_exposicao: float,
                                   b: float = B_SIGMOIDE,
                                   m: float = M_SIGMOIDE) -> float:
    """
    Converte exposição acumulada em conhecimento estimado (0 a 1)
    por meio de uma curva sigmoide.
    """
    xp_exposicao = float(xp_exposicao)
    return 1.0 / (1.0 + np.exp(-b * (xp_exposicao - m)))


def calcular_aprendizado_marginal(xp_exposicao: float,
                                  b: float = B_SIGMOIDE,
                                  m: float = M_SIGMOIDE) -> float:
    """
    Calcula o aprendizado marginal (derivada da sigmoide).
    Mede o quanto o analista ainda tende a aprender com mais experiência.
    """
    k = calcular_conhecimento_sigmoide(xp_exposicao, b=b, m=m)
    return float(b * k * (1.0 - k))


def inicializar_estado_aprendizagem(DB_AN: pd.DataFrame,
                                    df_grupos: pd.DataFrame,
                                    b: float = B_SIGMOIDE,
                                    m: float = M_SIGMOIDE) -> pd.DataFrame:
    """
    Cria a base de estado de aprendizagem com granularidade:
        analista × área

    Todas as combinações começam com exposição zero.
    """
    analistas = DB_AN["nome"].astype(str).unique().tolist()
    areas = df_grupos["Grupo"].astype(str).unique().tolist()

    grade = pd.MultiIndex.from_product(
        [analistas, areas],
        names=["analista", "area"]
    ).to_frame(index=False)

    grade["xp_exposicao"] = 0.0
    grade["conhecimento_k"] = grade["xp_exposicao"].apply(
        lambda x: calcular_conhecimento_sigmoide(x, b=b, m=m)
    )
    grade["aprendizado_marginal"] = grade["xp_exposicao"].apply(
        lambda x: calcular_aprendizado_marginal(x, b=b, m=m)
    )

    return grade


def atualizar_estado_aprendizagem(DB_WK_parcial: pd.DataFrame,
                                  estado_aprendizagem: pd.DataFrame,
                                  agora: pd.Timestamp,
                                  b: float = B_SIGMOIDE,
                                  m: float = M_SIGMOIDE):
    """
    Atualiza o estado de aprendizagem até o instante 'agora'.

    Regras:
    - só tarefas com fim_wk <= agora geram experiência;
    - cada tarefa só pode gerar experiência uma vez;
    - o crédito da experiência ocorre na área da própria tarefa.
    
    Retorna:
    - DB_WK_parcial atualizado (com xp_creditada ajustado)
    - estado_aprendizagem atualizado
    """
    if DB_WK_parcial.empty:
        return DB_WK_parcial, estado_aprendizagem

    base = DB_WK_parcial.copy()

    # garantir existência da coluna xp_creditada
    if "xp_creditada" not in base.columns:
        base["xp_creditada"] = False

    # selecionar tarefas concluídas até agora e ainda não creditadas
    mask_creditar = (
        (base["fim_wk"] <= agora) &
        (base["xp_creditada"] == False)
    )

    tarefas_para_creditar = base[mask_creditar].copy()

    if tarefas_para_creditar.empty:
        return base, estado_aprendizagem

    fator_escala_xp = 0.05
    tarefas_para_creditar["xp_tarefa"] = (
        tarefas_para_creditar["comp_est"].astype(float) *
        tarefas_para_creditar["tempo_est_h"].astype(float) *
        fator_escala_xp
    )

    # agregar xp por analista e área
    xp_agregado = (
        tarefas_para_creditar
        .groupby(["analista", "area_wk"], as_index=False)["xp_tarefa"]
        .sum()
        .rename(columns={"area_wk": "area"})
    )

    # somar a exposição no estado de aprendizagem
    estado = estado_aprendizagem.copy()

    estado = estado_aprendizagem.copy()
    
    estado = estado.merge(
        xp_agregado.rename(columns={"xp_tarefa": "xp_incremento"}),
        on=["analista", "area"],
        how="left"
    )
    
    estado["xp_incremento"] = estado["xp_incremento"].fillna(0.0)
    estado["xp_exposicao"] = estado["xp_exposicao"] + estado["xp_incremento"]
    estado = estado.drop(columns=["xp_incremento"])

    # recalcular conhecimento e aprendizado marginal
    estado["conhecimento_k"] = estado["xp_exposicao"].apply(
        lambda x: calcular_conhecimento_sigmoide(x, b=b, m=m)
    )
    estado["aprendizado_marginal"] = estado["xp_exposicao"].apply(
        lambda x: calcular_aprendizado_marginal(x, b=b, m=m)
    )

    # marcar tarefas como já creditadas
    base.loc[mask_creditar, "xp_creditada"] = True

    return base, estado



#%%
# =========================================================
# 12) Loop principal da simulação
# =========================================================

linhas = []
dias_uteis = gerar_dias_uteis(data_inicio, total_dias_uteis)
contador_tarefas = 0

# estado paralelo de aprendizagem:
# acompanha a experiência acumulada por analista × área
estado_aprendizagem = inicializar_estado_aprendizagem(
    DB_AN=DB_AN,
    df_grupos=df_grupos,
    b=B_SIGMOIDE,
    m=M_SIGMOIDE
)

for dia_base in dias_uteis:
    for j in range(tarefas_por_dia):
        if contador_tarefas >= total_tarefas:
            break

        # instante em que a demanda entra no sistema
        data_aloc = timestamp_do_dia_util(dia_base, j)
        
        # histórico parcial já construído até este ponto
        DB_WK_parcial = pd.DataFrame(linhas, columns=colunas_db_wk)
        
        # atualizar o estado de aprendizagem até este instante
        # (ainda sem influenciar a alocação)
        DB_WK_parcial, estado_aprendizagem = atualizar_estado_aprendizagem(
            DB_WK_parcial=DB_WK_parcial,
            estado_aprendizagem=estado_aprendizagem,
            agora=data_aloc,
            b=B_SIGMOIDE,
            m=M_SIGMOIDE
        )
        
        # refletir eventuais mudanças de xp_creditada no histórico em memória
        linhas = DB_WK_parcial.to_dict(orient="records")
        
        # (1) sortear grupo conforme distribuição
        grupo = rng.choice(grupos, p=probs)

        # (2) sortear tarefa dentro do grupo
        tarefa, comp_est, tempo_est_h = sortear_tarefa_no_grupo(DB_AT, grupo, rng)

        # (3) escolher analista pela regra atual
        analista = escolher_analista_AHP(
            DB_AN=DB_AN,
            DB_WK=DB_WK_parcial,
            grupo=grupo,
            tarefa=tarefa,
            comp_est=comp_est,
            tempo_est_h=tempo_est_h,
            agora=data_aloc
        )

        # (4) descobrir quando o analista poderá começar a tarefa
        inicio_real = proxima_disponibilidade_analista(
            DB_WK=DB_WK_parcial,
            analista=analista,
            agora=data_aloc
        )

        # (5) calcular o fim real estimado considerando jornada útil
        fim_real_estimado = calcular_fim_estimado(inicio_real, tempo_est_h)

        # (6) montar linha
        nova = {
            "area_wk": grupo,
            "tarefa_WK": tarefa,
            "descritivo": "",
            "analista": analista,
            "data_aloc": data_aloc,
            "inicio_wk": inicio_real,
            "fim_wk": fim_real_estimado,
            "comp_est": comp_est,
            "comp_real": np.nan,
            "tempo_est_h": tempo_est_h,
            "prazo_est": fim_real_estimado,  # mantido por compatibilidade
            "prazo_real": pd.NaT,
            "crit_aloc": "ESPECIALIZACAO + WORKLOAD_REMANESCENTE",
            "resum_exec": "",
            "data_envio": pd.NaT,
            "data_aval": pd.NaT,
            "retorno": "",
            "status": "",  # será calculado depois por snapshot
            "xp_creditada": False  # tarefa nasce sem ter gerado experiência
        }

        linhas.append(nova)
        contador_tarefas += 1
# %%


# =========================================================
# 13) Montar DB_WK final
# =========================================================
DB_WK = pd.DataFrame(linhas, columns=colunas_db_wk)

# garantir tipos de data
cols_datas = ["data_aloc", "inicio_wk", "fim_wk", "prazo_est", "prazo_real", "data_envio", "data_aval"]
for c in cols_datas:
    if c in DB_WK.columns:
        DB_WK[c] = pd.to_datetime(DB_WK[c], errors="coerce")

# garantir tipo lógico da coluna de controle de experiência
if "xp_creditada" in DB_WK.columns:
    DB_WK["xp_creditada"] = DB_WK["xp_creditada"].fillna(False).astype(bool)

# snapshot de status no instante da última alocação
momento_snapshot = DB_WK["data_aloc"].max()

DB_WK["status"] = DB_WK.apply(
    lambda row: classificar_status(row["inicio_wk"], row["fim_wk"], momento_snapshot),
    axis=1
)

# atualizar o estado de aprendizagem até o fim da simulação
# isso garante que o estoque final reflita todas as tarefas concluídas
if not DB_WK.empty:
    instante_final = DB_WK["fim_wk"].max()

    DB_WK, estado_aprendizagem = atualizar_estado_aprendizagem(
        DB_WK_parcial=DB_WK,
        estado_aprendizagem=estado_aprendizagem,
        agora=instante_final,
        b=B_SIGMOIDE,
        m=M_SIGMOIDE
    )

# workload remanescente por analista no snapshot final
workload_final = []
for nome in DB_AN["nome"].astype(str).tolist():
    wl = workload_remanescente_analista(DB_WK, nome, momento_snapshot)
    workload_final.append({
        "analista": nome,
        "workload_remanescente_h": wl
    })

df_workload_final = pd.DataFrame(workload_final).sort_values(
    by="workload_remanescente_h",
    ascending=True
)

# =========================================================
# 14) Checagens rápidas
# =========================================================
print("\nShape do DB_WK:", DB_WK.shape)

print("\nPrimeiras linhas do DB_WK:")
print(DB_WK.head())

print("\nÚltimas linhas do DB_WK:")
print(DB_WK.tail())

print("\nStatus no snapshot final:")
print(DB_WK["status"].value_counts(dropna=False))

print("\nWorkload remanescente por analista no snapshot final:")
print(df_workload_final)

print("\nQuantidade de tarefas por analista:")
print(DB_WK["analista"].value_counts())

print("\nHoras totais alocadas por analista:")
print(DB_WK.groupby("analista")["tempo_est_h"].sum().sort_values())

print("\nEstado final de aprendizagem:")
print(estado_aprendizagem.head())

print("\nMaior exposição acumulada por analista e área:")
print(
    estado_aprendizagem
    .sort_values("xp_exposicao", ascending=False)
    .head(15)
)

# %% Gráfico


# garantir datetime
DB_WK["data_aloc"] = pd.to_datetime(DB_WK["data_aloc"])
DB_WK["inicio_wk"] = pd.to_datetime(DB_WK["inicio_wk"])
DB_WK["fim_wk"] = pd.to_datetime(DB_WK["fim_wk"])

analistas = DB_WK["analista"].unique()

# eixo do tempo:
# usar todos os eventos relevantes para enxergar melhor a dinâmica
tempos = sorted(
    set(DB_WK["data_aloc"].dropna().tolist()) |
    set(DB_WK["inicio_wk"].dropna().tolist()) |
    set(DB_WK["fim_wk"].dropna().tolist())
)


def workload_at_time(df, analista, t):
    tarefas = df[df["analista"] == analista]
    wl = 0.0

    for _, row in tarefas.iterrows():
        data_aloc = row["data_aloc"]
        inicio = row["inicio_wk"]
        fim = row["fim_wk"]
        tempo = row["tempo_est_h"]

        # tarefa ainda não entrou no sistema
        if t < data_aloc:
            continue

        # tarefa já alocada, mas ainda em fila
        elif data_aloc <= t < inicio:
            wl += tempo

        # tarefa em execução
        elif inicio <= t < fim:
            wl += horas_uteis_entre(t, fim)

        # tarefa concluída
        else:
            continue

    return wl

# montar base do gráfico
dados_plot = []

for t in tempos:
    for a in analistas:
        wl = workload_at_time(DB_WK, a, t)
        dados_plot.append({
            "tempo": t,
            "analista": a,
            "workload": wl
        })

df_plot = pd.DataFrame(dados_plot)

# plot
plt.figure(figsize=(12, 7))

for a in analistas:
    sub = df_plot[df_plot["analista"] == a]
    plt.plot(sub["tempo"], sub["workload"], label=a)

plt.xlabel("Tempo")
plt.ylabel("Workload (horas remanescentes)")
plt.title("Workload ao longo do tempo por analista")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%  Extração de métricas da equipe

# =========================================================
# 15) Métricas a cada 10 dias úteis
# =========================================================

def status_tarefa_no_tempo(row: pd.Series, t: pd.Timestamp) -> str:
    """
    Classifica o status da tarefa no instante t.
    """
    data_aloc = row["data_aloc"]
    inicio = row["inicio_wk"]
    fim = row["fim_wk"]

    if t < data_aloc:
        return "NAO_EXISTIA"

    if data_aloc <= t < inicio:
        return "FILA"

    if inicio <= t < fim:
        return "EM_EXECUCAO"

    return "CONCLUIDA"


def workload_remanescente_tarefa_no_tempo(row: pd.Series, t: pd.Timestamp) -> float:
    """
    Horas remanescentes reais da tarefa no instante t.
    """
    data_aloc = row["data_aloc"]
    inicio = row["inicio_wk"]
    fim = row["fim_wk"]
    tempo = float(row["tempo_est_h"])

    if t < data_aloc:
        return 0.0

    if data_aloc <= t < inicio:
        return tempo

    if inicio <= t < fim:
        return horas_uteis_entre(t, fim)

    return 0.0


def fila_horas_tarefa_no_tempo(row: pd.Series, t: pd.Timestamp) -> float:
    """
    Tempo de fila acumulado da tarefa até o instante t, em horas úteis.
    """
    data_aloc = row["data_aloc"]
    inicio = row["inicio_wk"]

    if t < data_aloc:
        return 0.0

    if data_aloc <= t < inicio:
        return horas_uteis_entre(data_aloc, t)

    return horas_uteis_entre(data_aloc, inicio)


def gerar_snapshots_uteis(inicio: pd.Timestamp, fim: pd.Timestamp, passo_dias_uteis: int = 10):
    """
    Gera snapshots a cada N dias úteis, de 09:00.
    """
    dias_uteis = []
    atual = pd.Timestamp(inicio).normalize()

    while atual <= pd.Timestamp(fim).normalize():
        if eh_dia_util(atual):
            dias_uteis.append(atual)
        atual += pd.Timedelta(days=1)

    snapshots = []
    for i in range(0, len(dias_uteis), passo_dias_uteis):
        snapshots.append(dias_uteis[i] + pd.Timedelta(hours=9))

    # garantir que o último ponto da simulação entre como snapshot
    ultimo_ponto = ajustar_para_horario_util(fim)
    if len(snapshots) == 0 or snapshots[-1] != ultimo_ponto:
        snapshots.append(ultimo_ponto)

    return snapshots


def extrair_metricas_snapshot(DB_WK: pd.DataFrame, DB_AN: pd.DataFrame, t: pd.Timestamp) -> dict:
    """
    Extrai métricas do sistema no instante t.
    """
    base = DB_WK.copy()

    # status da tarefa no snapshot
    base["status_snapshot"] = base.apply(lambda row: status_tarefa_no_tempo(row, t), axis=1)

    # workload remanescente da tarefa no snapshot
    base["workload_rem_tarefa_h"] = base.apply(
        lambda row: workload_remanescente_tarefa_no_tempo(row, t), axis=1
    )

    # fila acumulada da tarefa no snapshot
    base["fila_tarefa_h"] = base.apply(
        lambda row: fila_horas_tarefa_no_tempo(row, t), axis=1
    )

    # workload por analista no snapshot
    workload_por_analista = (
        base.groupby("analista")["workload_rem_tarefa_h"]
        .sum()
        .reindex(DB_AN["nome"].astype(str).tolist(), fill_value=0.0)
    )

    # tarefas já existentes até t
    existentes = base[base["data_aloc"] <= t].copy()

    # concluídas até t
    concluidas = existentes[existentes["status_snapshot"] == "CONCLUIDA"].copy()

    # fila no instante t
    fila = existentes[existentes["status_snapshot"] == "FILA"].copy()

    # execução no instante t
    execucao = existentes[existentes["status_snapshot"] == "EM_EXECUCAO"].copy()

    # lead time das concluídas até t
    if not concluidas.empty:
        lead_time_h = (
            concluidas.apply(lambda row: horas_uteis_entre(row["data_aloc"], row["fim_wk"]), axis=1)
        )
        lead_time_medio_h = float(lead_time_h.mean())
    else:
        lead_time_medio_h = 0.0

    # percentual de analistas ociosos
    percentual_ociosos = float((workload_por_analista == 0).mean() * 100)

    metricas = {
        "data_snapshot": t,
        "n_analistas": int(DB_AN["nome"].nunique()),
        "tarefas_existentes": int(len(existentes)),
        "tarefas_concluidas": int(len(concluidas)),
        "tarefas_em_execucao": int(len(execucao)),
        "tarefas_em_fila": int(len(fila)),
        "workload_total_h": float(workload_por_analista.sum()),
        "workload_medio_por_analista_h": float(workload_por_analista.mean()),
        "workload_max_analista_h": float(workload_por_analista.max()),
        "percentual_analistas_ociosos": percentual_ociosos,
        "fila_media_h": float(fila["fila_tarefa_h"].mean()) if not fila.empty else 0.0,
        "fila_max_h": float(fila["fila_tarefa_h"].max()) if not fila.empty else 0.0,
        "lead_time_medio_concluidas_h": lead_time_medio_h,
    }

    return metricas


# =========================================================
# 16) Gerar df_metricas
# =========================================================

inicio_simulacao = DB_WK["data_aloc"].min()
fim_simulacao = DB_WK["fim_wk"].max()

snapshots = gerar_snapshots_uteis(
    inicio=inicio_simulacao,
    fim=fim_simulacao,
    passo_dias_uteis=10
)

metricas = []
for t in snapshots:
    metricas.append(extrair_metricas_snapshot(DB_WK, DB_AN, t))

df_metricas = pd.DataFrame(metricas)

print(df_metricas.head())
print(df_metricas.tail())

# %%
# =========================================================
# 17) Função para selecionar o time de cada cenário
# =========================================================

def selecionar_time(DB_AN_base: pd.DataFrame,
                    n_analistas: int,
                    criterio: str = "sample",
                    seed: int = 42) -> pd.DataFrame:
    """
    Seleciona um subconjunto de analistas para o cenário.

    Parâmetros:
    - DB_AN_base: base completa de analistas
    - n_analistas: tamanho do time no cenário
    - criterio:
        * 'sample'    -> sorteio aleatório reprodutível
        * 'primeiros' -> pega os primeiros N da base
    - seed: semente para reprodutibilidade do sorteio
    """
    if n_analistas > len(DB_AN_base):
        raise ValueError("n_analistas é maior do que a quantidade disponível em DB_AN_base.")

    base = DB_AN_base.copy()

    if criterio == "sample":
        return (
            base
            .sample(n=n_analistas, random_state=seed)
            .copy()
            .reset_index(drop=True)
        )

    if criterio == "primeiros":
        return (
            base
            .head(n_analistas)
            .copy()
            .reset_index(drop=True)
        )

    raise ValueError("Critério inválido. Use 'sample' ou 'primeiros'.")


# =========================================================
# 17-A) Função para montar distribuições de demanda por cenário
# =========================================================

def montar_distribuicao_cenario(df_grupos_base: pd.DataFrame,
                                tipo_cenario: str) -> pd.DataFrame:
    """
    Retorna um dataframe de distribuição de grupos para o cenário.

    Parâmetros:
    - df_grupos_base: dataframe original com colunas Grupo, Distribuição e p
    - tipo_cenario:
        * "base"
        * "concentrado_societario"

    Regras:
    - cenário base: mantém a distribuição original
    - cenário concentrado_societario:
        50% da probabilidade vai para Societário
        os outros 50% são redistribuídos proporcionalmente entre as demais áreas
    """
    dist = df_grupos_base.copy()

    # garantir cópia independente
    dist = dist[["Grupo", "Distribuição", "p"]].copy()

    if tipo_cenario == "base":
        dist["p_cenario"] = dist["p"]
        return dist

    if tipo_cenario == "concentrado_societario":
        alvo = "Societário"

        if alvo not in dist["Grupo"].values:
            raise ValueError("A área 'Societário' não foi encontrada em df_grupos.")

        # separar a linha-alvo das demais
        mask_alvo = dist["Grupo"] == alvo
        dist["p_cenario"] = 0.0

        # 50% fixos para Societário
        dist.loc[mask_alvo, "p_cenario"] = 0.50

        # os outros 50% são redistribuídos entre as demais áreas,
        # preservando a proporcionalidade original
        outras = dist.loc[~mask_alvo].copy()

        soma_outras = outras["p"].sum()
        if soma_outras <= 0:
            raise ValueError("A soma das probabilidades das outras áreas é inválida.")

        dist.loc[~mask_alvo, "p_cenario"] = 0.50 * (dist.loc[~mask_alvo, "p"] / soma_outras)

        # pequena proteção numérica
        dist["p_cenario"] = dist["p_cenario"] / dist["p_cenario"].sum()

        return dist

    raise ValueError("tipo_cenario inválido. Use 'base' ou 'concentrado_societario'.")

# =========================================================
# 18) Função para rodar a simulação de um cenário
# =========================================================
def rodar_simulacao(DB_AN_cenario: pd.DataFrame,
                    DB_AT: pd.DataFrame,
                    df_grupos: pd.DataFrame,
                    total_tarefas: int = 1000,
                    tarefas_por_dia: int = 5,
                    data_inicio: str = "2026-01-01",
                    seed_sim: int = 123,
                    distribuicao_cenario: pd.DataFrame = None):
    """
    Versão otimizada do motor de simulação.

    Melhorias:
    - não reconstrói DB_WK dentro do loop;
    - não recalcula workload histórico a cada tarefa;
    - usa estado incremental de disponibilidade por analista;
    - faz sorteio de tarefas com base pré-processada por área.
    """
    rng = np.random.default_rng(seed_sim)

    data_inicio = pd.Timestamp(data_inicio)
    total_dias_uteis = math.ceil(total_tarefas / tarefas_por_dia)

    if distribuicao_cenario is None:
        grupos = df_grupos["Grupo"].astype(str).to_list()
        probs = df_grupos["p"].astype(float).to_list()
    else:
        grupos = distribuicao_cenario["Grupo"].astype(str).to_list()
        probs = distribuicao_cenario["p_cenario"].astype(float).to_list()

    tarefas_area = preparar_tarefas_por_area(DB_AT)

    linhas = []
    dias_uteis = gerar_dias_uteis(data_inicio, total_dias_uteis)
    contador_tarefas = 0

    estado_analistas = inicializar_estado_analistas(
        DB_AN_cenario=DB_AN_cenario,
        data_inicio=data_inicio
    )

    for dia_base in dias_uteis:
        for j in range(tarefas_por_dia):
            if contador_tarefas >= total_tarefas:
                break

            data_aloc = timestamp_do_dia_util(dia_base, j)

            grupo = rng.choice(grupos, p=probs)

            tarefa, comp_est, tempo_est_h = sortear_tarefa_no_grupo_rapido(
                tarefas_area=tarefas_area,
                grupo=grupo,
                rng=rng
            )

            analista = escolher_analista_rapido(
                estado_analistas=estado_analistas,
                grupo=grupo
            )

            idx = estado_analistas.index[
                estado_analistas["analista"] == analista
            ][0]

            inicio_real = ajustar_para_horario_util(
                max(data_aloc, estado_analistas.loc[idx, "disponivel_em"])
            )

            fim_real_estimado = calcular_fim_estimado(inicio_real, tempo_est_h)

            estado_analistas.loc[idx, "disponivel_em"] = fim_real_estimado

            linhas.append({
                "area_wk": grupo,
                "tarefa_WK": tarefa,
                "descritivo": "",
                "analista": analista,
                "data_aloc": data_aloc,
                "inicio_wk": inicio_real,
                "fim_wk": fim_real_estimado,
                "comp_est": comp_est,
                "comp_real": np.nan,
                "tempo_est_h": tempo_est_h,
                "prazo_est": fim_real_estimado,
                "prazo_real": pd.NaT,
                "crit_aloc": "ESPECIALIZACAO + DISPONIBILIDADE",
                "resum_exec": "",
                "data_envio": pd.NaT,
                "data_aval": pd.NaT,
                "retorno": "",
                "status": "",
                "xp_creditada": False
            })

            contador_tarefas += 1

    DB_WK_cenario = pd.DataFrame(linhas, columns=colunas_db_wk)

    cols_datas = [
        "data_aloc", "inicio_wk", "fim_wk",
        "prazo_est", "prazo_real", "data_envio", "data_aval"
    ]
    for c in cols_datas:
        if c in DB_WK_cenario.columns:
            DB_WK_cenario[c] = pd.to_datetime(DB_WK_cenario[c], errors="coerce")

    if "xp_creditada" in DB_WK_cenario.columns:
        DB_WK_cenario["xp_creditada"] = DB_WK_cenario["xp_creditada"].fillna(False).astype(bool)

    momento_snapshot = DB_WK_cenario["data_aloc"].max()

    DB_WK_cenario["status"] = DB_WK_cenario.apply(
        lambda row: classificar_status(row["inicio_wk"], row["fim_wk"], momento_snapshot),
        axis=1
    )

    estado_aprendizagem_cenario = inicializar_estado_aprendizagem(
        DB_AN=DB_AN_cenario,
        df_grupos=df_grupos,
        b=B_SIGMOIDE,
        m=M_SIGMOIDE
    )

    if not DB_WK_cenario.empty:
        instante_final = DB_WK_cenario["fim_wk"].max()

        DB_WK_cenario, estado_aprendizagem_cenario = atualizar_estado_aprendizagem(
            DB_WK_parcial=DB_WK_cenario,
            estado_aprendizagem=estado_aprendizagem_cenario,
            agora=instante_final,
            b=B_SIGMOIDE,
            m=M_SIGMOIDE
        )

    return DB_WK_cenario, estado_aprendizagem_cenario

# =========================================================
# 19) Função para gerar métricas de um cenário
# =========================================================

def gerar_metricas_cenario(DB_WK_cenario: pd.DataFrame,
                           DB_AN_cenario: pd.DataFrame,
                           passo_dias_uteis: int = 10) -> pd.DataFrame:
    """
    Gera o dataframe de métricas do cenário em snapshots periódicos.
    """
    inicio_simulacao = DB_WK_cenario["data_aloc"].min()
    fim_simulacao = DB_WK_cenario["fim_wk"].max()

    snapshots = gerar_snapshots_uteis(
        inicio=inicio_simulacao,
        fim=fim_simulacao,
        passo_dias_uteis=passo_dias_uteis
    )

    metricas = []
    for t in snapshots:
        metricas.append(
            extrair_metricas_snapshot(DB_WK_cenario, DB_AN_cenario, t)
        )

    return pd.DataFrame(metricas)


# =========================================================
# 19-A) Funções de conhecimento e concentração
# =========================================================

def extrair_conhecimento_snapshot(DB_WK: pd.DataFrame,
                                  DB_AN: pd.DataFrame,
                                  df_grupos: pd.DataFrame,
                                  t: pd.Timestamp,
                                  b: float = B_SIGMOIDE,
                                  m: float = M_SIGMOIDE) -> pd.DataFrame:
    """
    Gera a base analítica analista × área em um dado snapshot.

    Regras:
    - só entram tarefas concluídas até o snapshot (fim_wk <= t);
    - a exposição prática da tarefa é dada por:
          xp_tarefa = comp_est × tempo_est_h
    - o conhecimento é calculado pela curva sigmoide;
    - o share de conhecimento é calculado dentro de cada área;
    - a contribuição ao HHI é share².
    """

    # -----------------------------------------------------
    # 1) Selecionar tarefas concluídas até o snapshot
    # -----------------------------------------------------
    concluidas = DB_WK[DB_WK["fim_wk"] <= t].copy()

    # -----------------------------------------------------
    # 2) Calcular xp da tarefa
    # -----------------------------------------------------
    if not concluidas.empty:
        concluidas["xp_tarefa"] = concluidas.apply(
            lambda row: calcular_xp_tarefa(row["comp_est"], row["tempo_est_h"]),
            axis=1
        )

        xp_agregado = (
            concluidas
            .groupby(["analista", "area_wk"], as_index=False)["xp_tarefa"]
            .sum()
            .rename(columns={
                "area_wk": "area",
                "xp_tarefa": "xp_exposicao"
            })
        )
    else:
        xp_agregado = pd.DataFrame(columns=["analista", "area", "xp_exposicao"])

    # -----------------------------------------------------
    # 3) Criar grade completa analista × área
    # -----------------------------------------------------
    analistas = DB_AN["nome"].astype(str).unique().tolist()
    areas = df_grupos["Grupo"].astype(str).unique().tolist()

    grade = pd.MultiIndex.from_product(
        [analistas, areas],
        names=["analista", "area"]
    ).to_frame(index=False)

    # -----------------------------------------------------
    # 4) Fazer merge da grade com a exposição agregada
    # -----------------------------------------------------
    base = grade.merge(
        xp_agregado,
        on=["analista", "area"],
        how="left"
    )

    base["xp_exposicao"] = base["xp_exposicao"].fillna(0.0)

    # -----------------------------------------------------
    # 5) Calcular conhecimento e aprendizado marginal
    # -----------------------------------------------------
    base["conhecimento_k"] = base["xp_exposicao"].apply(
        lambda x: calcular_conhecimento_sigmoide(x, b=b, m=m)
    )

    base["aprendizado_marginal"] = base["xp_exposicao"].apply(
        lambda x: calcular_aprendizado_marginal(x, b=b, m=m)
    )

    # -----------------------------------------------------
    # 6) Calcular participação relativa no conhecimento da área
    # -----------------------------------------------------
    total_por_area = (
        base.groupby("area")["conhecimento_k"]
        .sum()
        .rename("conhecimento_total_area")
        .reset_index()
    )

    base = base.merge(total_por_area, on="area", how="left")

    # evitar divisão por zero
    base["share_conhecimento"] = np.where(
        base["conhecimento_total_area"] > 0,
        base["conhecimento_k"] / base["conhecimento_total_area"],
        0.0
    )

    # contribuição individual para o HHI
    base["contribuicao_hhi"] = base["share_conhecimento"] ** 2

    # registrar o snapshot
    base["data_snapshot"] = t

    # organizar colunas
    base = base[
        [
            "data_snapshot",
            "analista",
            "area",
            "xp_exposicao",
            "conhecimento_k",
            "aprendizado_marginal",
            "conhecimento_total_area",
            "share_conhecimento",
            "contribuicao_hhi"
        ]
    ].copy()

    return base


def extrair_concentracao_snapshot(df_conhecimento_snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida a base analista × área em uma base por área,
    calculando métricas de concentração de conhecimento.
    """

    resultados = []

    for area, sub in df_conhecimento_snapshot.groupby("area"):
        sub = sub.sort_values("share_conhecimento", ascending=False).reset_index(drop=True)

        data_snapshot = sub["data_snapshot"].iloc[0]
        conhecimento_total_area = float(sub["conhecimento_total_area"].iloc[0])
        hhi_conhecimento = float(sub["contribuicao_hhi"].sum())

        share_top1 = float(sub["share_conhecimento"].iloc[0]) if len(sub) >= 1 else 0.0
        share_top2 = float(sub["share_conhecimento"].iloc[:2].sum()) if len(sub) >= 2 else share_top1

        analista_top1 = sub["analista"].iloc[0] if len(sub) >= 1 else None
        analista_top2 = sub["analista"].iloc[1] if len(sub) >= 2 else None

        n_analistas_com_conhecimento = int((sub["conhecimento_k"] > 0).sum())

        resultados.append({
            "data_snapshot": data_snapshot,
            "area": area,
            "conhecimento_total_area": conhecimento_total_area,
            "hhi_conhecimento": hhi_conhecimento,
            "share_top1": share_top1,
            "share_top2": share_top2,
            "analista_top1": analista_top1,
            "analista_top2": analista_top2,
            "n_analistas_com_conhecimento": n_analistas_com_conhecimento
        })

    return pd.DataFrame(resultados)


def gerar_conhecimento_e_concentracao_cenario(DB_WK_cenario: pd.DataFrame,
                                              DB_AN_cenario: pd.DataFrame,
                                              df_grupos: pd.DataFrame,
                                              passo_dias_uteis: int = 10,
                                              b: float = B_SIGMOIDE,
                                              m: float = M_SIGMOIDE):
    """
    Gera, para um cenário, as duas bases:
    - df_conhecimento_analista_area
    - df_concentracao_area
    """

    inicio_simulacao = DB_WK_cenario["data_aloc"].min()
    fim_simulacao = DB_WK_cenario["fim_wk"].max()

    snapshots = gerar_snapshots_uteis(
        inicio=inicio_simulacao,
        fim=fim_simulacao,
        passo_dias_uteis=passo_dias_uteis
    )

    lista_conhecimento = []
    lista_concentracao = []

    for t in snapshots:
        # base detalhada analista × área
        df_conhecimento_snapshot = extrair_conhecimento_snapshot(
            DB_WK=DB_WK_cenario,
            DB_AN=DB_AN_cenario,
            df_grupos=df_grupos,
            t=t,
            b=b,
            m=m
        )

        # base consolidada por área
        df_concentracao_snapshot = extrair_concentracao_snapshot(
            df_conhecimento_snapshot=df_conhecimento_snapshot
        )

        lista_conhecimento.append(df_conhecimento_snapshot)
        lista_concentracao.append(df_concentracao_snapshot)

    df_conhecimento_analista_area = pd.concat(lista_conhecimento, ignore_index=True)
    df_concentracao_area = pd.concat(lista_concentracao, ignore_index=True)

    return df_conhecimento_analista_area, df_concentracao_area

# =========================================================
# 20) Parâmetros gerais dos cenários operacionais
# =========================================================

DB_AN_base_cenarios = DB_AN.copy()

config_cenarios = {
    "cenario_1_base": {
        "tipo_distribuicao": "base",
        "tarefas_por_dia": 5,
        "total_tarefas": 1000
    },
    "cenario_2_concentrado_societario": {
        "tipo_distribuicao": "concentrado_societario",
        "tarefas_por_dia": 5,
        "total_tarefas": 1000
    },
    "cenario_3_sobrecarga": {
        "tipo_distribuicao": "base",
        "tarefas_por_dia": 15,
        "total_tarefas": 1000
    },
    "cenario_4_concentrado_societario_sobrecarga": {
        "tipo_distribuicao": "concentrado_societario",
        "tarefas_por_dia": 15,
        "total_tarefas": 1000
    }
}

tamanhos_time = [5, 7, 10]
seed_simulacao = 123
passo_metricas = 20

# dicionário final: cada chave = uma execução
resultados = {}

# =========================================================
# 21) Rodar cenários operacionais
# =========================================================

for nome_cenario, config in config_cenarios.items():
    for n_analistas in tamanhos_time:

        print(f"\nRodando {nome_cenario} com {n_analistas} analistas")

        # 1) Selecionar subconjunto de analistas
        DB_AN_cenario = selecionar_time(
            DB_AN_base=DB_AN_base_cenarios,
            n_analistas=n_analistas,
            criterio="sample",
            seed=42
        )

        # 2) Montar distribuição do cenário
        distribuicao_cenario = montar_distribuicao_cenario(
            df_grupos_base=df_grupos,
            tipo_cenario=config["tipo_distribuicao"]
        )

        # 3) Rodar simulação
        DB_WK_cenario, estado_aprendizagem_cenario = rodar_simulacao(
            DB_AN_cenario=DB_AN_cenario,
            DB_AT=DB_AT,
            df_grupos=df_grupos,
            total_tarefas=config["total_tarefas"],
            tarefas_por_dia=config["tarefas_por_dia"],
            data_inicio="2026-01-01",
            seed_sim=seed_simulacao,
            distribuicao_cenario=distribuicao_cenario
        )

        # 4) Métricas operacionais
        df_metricas_cenario = gerar_metricas_cenario(
            DB_WK_cenario=DB_WK_cenario,
            DB_AN_cenario=DB_AN_cenario,
            passo_dias_uteis=passo_metricas
        )

        # 5) Conhecimento e concentração
        df_conhecimento_cenario, df_concentracao_cenario = gerar_conhecimento_e_concentracao_cenario(
            DB_WK_cenario=DB_WK_cenario,
            DB_AN_cenario=DB_AN_cenario,
            df_grupos=df_grupos,
            passo_dias_uteis=passo_metricas,
            b=B_SIGMOIDE,
            m=M_SIGMOIDE
        )

        # 6) Identificação da execução
        chave = f"{nome_cenario}_{n_analistas}_analistas"

        df_metricas_cenario["cenario"] = nome_cenario
        df_metricas_cenario["n_analistas_cenario"] = n_analistas
        df_metricas_cenario["execucao"] = chave

        df_conhecimento_cenario["cenario"] = nome_cenario
        df_conhecimento_cenario["n_analistas_cenario"] = n_analistas
        df_conhecimento_cenario["execucao"] = chave

        df_concentracao_cenario["cenario"] = nome_cenario
        df_concentracao_cenario["n_analistas_cenario"] = n_analistas
        df_concentracao_cenario["execucao"] = chave

        # 7) Salvar resultados
        resultados[chave] = {
            "cenario": nome_cenario,
            "n_analistas": n_analistas,
            "DB_AN": DB_AN_cenario.copy(),
            "distribuicao_cenario": distribuicao_cenario.copy(),
            "DB_WK": DB_WK_cenario.copy(),
            "estado_aprendizagem": estado_aprendizagem_cenario.copy(),
            "df_metricas": df_metricas_cenario.copy(),
            "df_conhecimento_analista_area": df_conhecimento_cenario.copy(),
            "df_concentracao_area": df_concentracao_cenario.copy()
        }

        print("Execução concluída:", chave)

# =========================================================
# 22) Consolidar métricas de todas as execuções
# =========================================================

if not resultados:
    raise ValueError("Nenhuma execução foi gerada. Verifique o loop de cenários.")

lista_metricas = [conteudo["df_metricas"] for conteudo in resultados.values()]
df_metricas_cenarios = pd.concat(lista_metricas, ignore_index=True)

lista_conhecimento = [conteudo["df_conhecimento_analista_area"] for conteudo in resultados.values()]
df_conhecimento_cenarios = pd.concat(lista_conhecimento, ignore_index=True)

lista_concentracao = [conteudo["df_concentracao_area"] for conteudo in resultados.values()]
df_concentracao_cenarios = pd.concat(lista_concentracao, ignore_index=True)

print("\nMétricas consolidadas:")
print(df_metricas_cenarios.head())

print("\nConhecimento consolidado:")
print(df_conhecimento_cenarios.head())

print("\nConcentração consolidada:")
print(df_concentracao_cenarios.head())

# =========================================================
# 23) Função auxiliar para plotar comparações entre cenários
# =========================================================


def plotar_cenarios(df_metricas: pd.DataFrame,
                    coluna_y: str,
                    titulo: str,
                    ylabel: str):
    """
    Gera um gráfico de linhas comparando cenários ao longo do tempo.
    """
    df_plot = df_metricas.copy().sort_values("data_snapshot")

    plt.figure(figsize=(10, 6))

    for cenario in sorted(df_plot["cenario"].unique()):
        sub = df_plot[df_plot["cenario"] == cenario]
        plt.plot(sub["data_snapshot"], sub[coluna_y], label=cenario)

    plt.title(titulo)
    plt.xlabel("Tempo")
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =========================================================
# 24) Gráficos comparativos dos cenários
# =========================================================

# 1) workload total do time
plotar_cenarios(
    df_metricas=df_metricas_cenarios,
    coluna_y="workload_total_h",
    titulo="Workload total do time ao longo do tempo",
    ylabel="Workload total (horas)"
)

# 2) tarefas em fila
plotar_cenarios(
    df_metricas=df_metricas_cenarios,
    coluna_y="tarefas_em_fila",
    titulo="Quantidade de tarefas em fila",
    ylabel="Tarefas em fila"
)

# 3) percentual de ociosidade
plotar_cenarios(
    df_metricas=df_metricas_cenarios,
    coluna_y="percentual_analistas_ociosos",
    titulo="Percentual de analistas ociosos",
    ylabel="% de ociosidade"
)

# 4) lead time médio das tarefas concluídas
plotar_cenarios(
    df_metricas=df_metricas_cenarios,
    coluna_y="lead_time_medio_concluidas_h",
    titulo="Lead time médio das tarefas concluídas",
    ylabel="Horas"
)

# %%

# =========================================================
# 25) Exportar resultados dos cenários
# =========================================================
import os

pasta_saida = REPO_ROOT / "results" / "without_AHP"
os.makedirs(pasta_saida, exist_ok=True)

metodo = "sem_AHP"

for nome_execucao, conteudo in resultados.items():
    pasta_execucao = os.path.join(pasta_saida, f"{metodo}_{nome_execucao}")
    os.makedirs(pasta_execucao, exist_ok=True)

    conteudo["DB_WK"].to_csv(
        os.path.join(pasta_execucao, "DB_WK.csv"),
        index=False, encoding="utf-8-sig", sep=";", decimal=","
    )

    conteudo["estado_aprendizagem"].to_csv(
        os.path.join(pasta_execucao, "estado_aprendizagem.csv"),
        index=False, encoding="utf-8-sig", sep=";", decimal=","
    )

    conteudo["df_metricas"].to_csv(
        os.path.join(pasta_execucao, "df_metricas.csv"),
        index=False, encoding="utf-8-sig", sep=";", decimal=","
    )

    conteudo["df_conhecimento_analista_area"].to_csv(
        os.path.join(pasta_execucao, "df_conhecimento_analista_area.csv"),
        index=False, encoding="utf-8-sig", sep=";", decimal=","
    )

    conteudo["df_concentracao_area"].to_csv(
        os.path.join(pasta_execucao, "df_concentracao_area.csv"),
        index=False, encoding="utf-8-sig", sep=";", decimal=","
    )

print(f"\nArquivos exportados com sucesso em:\n{pasta_saida}")
