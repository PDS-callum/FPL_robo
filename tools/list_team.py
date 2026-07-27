import requests


def main() -> None:
    bootstrap = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=30
    ).json()
    players = {p["id"]: p for p in bootstrap["elements"]}
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    resp = requests.get(
        "https://fantasy.premierleague.com/api/entry/1687992/event/1/picks/", timeout=30
    )
    print("status", resp.status_code)
    resp.raise_for_status()
    picks = resp.json().get("picks", [])

    for p in sorted(picks, key=lambda x: x["position"]):
        pl = players.get(p["element"], {})
        role = ""
        if p.get("is_captain"):
            role = " (C)"
        elif p.get("is_vice_captain"):
            role = " (VC)"
        print(
            f"{p['position']:>2}. {pos_map.get(pl.get('element_type'), '?'):<3} "
            f"{pl.get('web_name', p['element']):<18} {teams.get(pl.get('team'), '?'):<4} "
            f"£{(pl.get('now_cost', 0)/10):.1f}m{role}"
        )


if __name__ == "__main__":
    main()
