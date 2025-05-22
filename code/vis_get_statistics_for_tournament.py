import requests
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd

def process_tournament_statistics(tournament_no):
    # Define the API endpoint
    endpoint = "https://www.fivb.org/vis2009/XmlRequest.asmx"

    # Step i: Get a list of all matches in the tournament
    GetMatchRequest = f"""
    <Request Type="GetBeachMatchList" Fields="NoInTournament LocalDate LocalTime TeamAName TeamBName Court MatchPointsA MatchPointsB PointsTeamASet1 PointsTeamBSet1 PointsTeamASet2 PointsTeamBSet2 PointsTeamASet3 PointsTeamBSet3 DurationSet1 DurationSet2 DurationSet3">
        <Filter NoTournament="{tournament_no}" InMainDraw="true"/>
    </Request>
    """
    encoded_request = urllib.parse.quote(GetMatchRequest)
    url = f"{endpoint}?Request={encoded_request}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to fetch matches. HTTP Status Code: {response.status_code}")
        return None

    root = ET.fromstring(response.text)
    matches = []
    column_names = []
    first_match = root.find(".//BeachMatch")
    if first_match is not None:
        column_names = list(first_match.attrib.keys())
        for match in root.findall(".//BeachMatch"):
            row = {col: match.attrib.get(col, "") for col in column_names}
            matches.append(row)
    else:
        print("No matches found in the tournament.")
        return None

    matches_df = pd.DataFrame(matches)

    # Step ii: Get statistics for all matches
    match_statistics = []
    for match_no in matches_df["No"].unique():
        MatchStatisticRequest = f"""
        <Requests>
            <Request Type='GetBeachStatisticList' Fields='ItemType NoItem NoSet SpikeFault SpikePoint ServeFault ServePoint'>
                <Filter Type='VolleyStatisticFilter' NoMatches='{match_no}' />
            </Request>
        </Requests>
        """
        encoded_request = urllib.parse.quote(MatchStatisticRequest)
        url = f"{endpoint}?Request={encoded_request}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to fetch statistics for match {match_no}. HTTP Status Code: {response.status_code}")
            continue

        root = ET.fromstring(response.text)
        first_stat = root.find(".//VolleyStatistic")
        if first_stat is not None:
            stat_columns = list(first_stat.attrib.keys())
            for stat in root.findall(".//VolleyStatistic"):
                row = {col: stat.attrib.get(col, "") for col in stat_columns}
                match_statistics.append(row)

    statistics_df = pd.DataFrame(match_statistics)

    # Step iii: Retrieve a list of all players and teams
    unique_items = statistics_df["NoItem"].unique()
    players_teams = []
    for item_no in unique_items:
        GetPlayerRequest = f"""
        <Request Type="GetPlayer" No="{item_no}" Fields="FederationCode FirstName Gender LastName Nationality PlaysBeach PlaysVolley TeamName" />
        """
        encoded_request = urllib.parse.quote(GetPlayerRequest)
        url = f"{endpoint}?Request={encoded_request}"
        response = requests.get(url)

        if response.status_code == 200:
            root = ET.fromstring(response.text)
            if root.attrib:
                players_teams.append(root.attrib)
        else:
            GetTeamRequest = f"""
            <Request Type="GetBeachTeam" No="{item_no}" Fields="NoPlayer1 NoPlayer2 Name Rank EarnedPointsTeam" />
            """
            encoded_request = urllib.parse.quote(GetTeamRequest)
            url = f"{endpoint}?Request={encoded_request}"
            response = requests.get(url)

            if response.status_code == 200:
                root = ET.fromstring(response.text)
                if root.attrib:
                    players_teams.append(root.attrib)

    players_teams_df = pd.DataFrame(players_teams)

    # Step iv: Match the player/team dataframe to the statistics dataframe
    merged_df = statistics_df.merge(players_teams_df, left_on="NoItem", right_on="No", how="left")

    return merged_df

tournament_statistics = process_tournament_statistics(7565)
print(tournament_statistics)
if tournament_statistics is not None:
    # Write the DataFrame to a CSV file
    tournament_statistics.to_csv("tournament_statistics.csv", index=False, sep="\t", quoting=1)  # quoting=1 ensures fields are enclosed in quotes
    print("Tournament statistics saved to 'tournament_statistics.csv'.")
else:
    print("No tournament statistics to save.")