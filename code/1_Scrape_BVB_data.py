
# # Scraping the Beachvolleyball Data from fivb-vis

# %%
import xmltodict
from fivbvis import Beach
import requests
from fivbvis import Article
import pandas as pd
from fivbvis import Player
import xml.etree.ElementTree as ET
import urllib.parse


# ### List of Beachvolleyball Matches

# %%
def process_BeachMatchList():
    b = Beach()
    BeachMatchList = b.getBeachMatchList(fields="NoInTournament NoTournament LocalDate LocalTime TeamAName TeamBName Court MatchPointsA MatchPointsB PointsTeamASet1 PointsTeamBSet1 PointsTeamASet2 PointsTeamBSet2 PointsTeamASet3 PointsTeamBSet3 DurationSet1 DurationSet2 DurationSet3 NoPlayerA1 NoPlayerA2 NoPlayerB1 NoPlayerB2",
                                     )
    MatchList_dict = xmltodict.parse(BeachMatchList)
    beachMatch_list_df = pd.DataFrame(MatchList_dict['BeachMatches']['BeachMatch'])
    beachMatch_list_df.to_csv('BeachMatchList_full.csv', index=False, sep=';', quoting=1, encoding='UTF-8')
    return beachMatch_list_df


# %%
test = process_BeachMatchList()


# ### Beachvolleyball Tournaments

# %%
def getBeachTournament2(self, no, fields=None, content_type='xml'):
    result = self.fivb_vis.get('GetBeachTournament', no, fields, content_type)
    return result

#zuweisen
b = Beach()
b.getBeachTournament = getBeachTournament2

# %%
def process_tournaments(tournament_nos):
    tournament_list = []

    for no in tournament_nos:
        try:
            # API-Abfrage
            response = b.getBeachTournament(self=b, no=no, fields='')
            tournament_data = xmltodict.parse(response)

            # Extrahiere Attribute und entferne das @
            data = tournament_data['BeachTournament']
            clean_data = {k.lstrip("@"): v for k, v in data.items()}

            # Zur Gesamtliste hinzufügen
            tournament_list.append(clean_data)

        except Exception as e:
            print(f"Fehler bei Turnier-Nr. {no}: {e}")

    # Alles in einen DataFrame packen
    df = pd.DataFrame(tournament_list)

# %%
# Extract numbers of tournaments

tournament_nos = list(test['@NoTournament'].unique())
Tournaments_data = process_tournaments(tournament_nos)


# ### Beachvolleyball Statistic Data

# %%
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


# ### Data of Beachvolleyball Player

# %%
#anpassen woher daten id für player kommen

# %%
def process_BeachTeams():
    data_player_list = b.getBeachTeamList(self=b, fields = 'NoPlayer1 Player1TeamName NoPlayer2 Player2TeamName TournamentName TournamentEndDateMainDraw TournamentType Rank EarnedPointsTeam EarningsTotalTeam')
    player_list_dict = xmltodict.parse(data_player_list)
    player_list_df = pd.DataFrame(player_list_dict['BeachTeams']['BeachTeam'])
    return player_list_df

# %%
player_list_df = process_BeachTeams()

# %%
#Extract numbers of players

#Player 1
player1_list = player_list_df[(player_list_df['@NoPlayer1'].astype(str).str.len()>3)]
player1_list1 = player1_list[['@NoPlayer1','@Player1TeamName']].drop_duplicates()

#Player 2
player2_list = player_list_df[(player_list_df['@NoPlayer2'].astype(str).str.len()>3)]
player2_list2 = player2_list[['@NoPlayer2','@Player2TeamName']].drop_duplicates()

player_list_df2_test_err = player_list_df[(player_list_df['@NoPlayer2'].astype(str).str.len()<3)]
player_list_df2_test_err['@NoPlayer2'].unique()

player1_df = player1_list1[['@NoPlayer1', '@Player1TeamName']].rename(
    columns={'@NoPlayer1': 'PlayerId', '@Player1TeamName': 'PlayerName'}
)

player2_df = player2_list2[['@NoPlayer2', '@Player2TeamName']].rename(
    columns={'@NoPlayer2': 'PlayerId', '@Player2TeamName': 'PlayerName'}
)

player_list_unique = pd.concat([player1_df, player2_df], ignore_index=True).astype(str).drop_duplicates()


# CSV schreiben. Liste enthält nur ID und Nachname unique
# player_list_unique.to_csv('player_list_NO_Name.csv', index=False, sep=';', quoting=1, encoding='UTF-8')

# # CSV schreiben: kompletter datensatz mit allen variablen
# player_list_df.to_csv('player_list.csv', index=False, sep=';', quoting=1, encoding='UTF-8')

# %%
def process_PlayerInfo(no):
    player = Player()
    Player_full_list = []
    for playerNo in player_list_unique['ModeNoPlayer2']:
        print(playerNo)
        Info = player.getPlayerInfo(no=playerNo,
                                 fields='FederationCode FirstName Gender LastName Nationality PlaysBeach PlaysVolley TeamName Height BeachCurrentTeam BeachHighBlock BeachHighJump BeachPosition BeachRetiredDate BeachYearBegin Birthdate Handedness PlayerStatus PlayerType')
        Info_dict = xmltodict.parse(Info)
        list_info = pd.json_normalize(Info_dict)
        Player_full_list.append(list_info)
    Player_full_df = pd.concat(Player_full_list, ignore_index=True)
    return Player_full_df




