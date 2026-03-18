# generate_params.py
from config import config

def main():
    conf = config()
    p_list = conf['p_list']
    alpha_list = conf['alpha_list']
    N_list = conf['N_list']

    with open("params.txt", "w") as f:
        for p in p_list:
            for alpha in alpha_list:
                for N in N_list:
                    # Strictly comma-separated, NO spaces after the comma
                    f.write(f"{N},{p:.3f},{alpha:.4f}\n")

    print(f"Generated params.txt with {len(N_list) * len(p_list) * len(alpha_list)} jobs.")

if __name__ == "__main__":
    main()
