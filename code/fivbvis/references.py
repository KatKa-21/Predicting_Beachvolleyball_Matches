from .fivbvis import FivbVis

class Article(FivbVis):
    def __init__(self):
        self.fivb_vis = FivbVis()

    def getArticle(self, no, fields=None, content_type='xml'):
        result = self.fivb_vis.get('GetArticle', no, fields, content_type)
        return result

    def getArticleListWithFilter(self, fields=None, filter=None, content_type='xml'):
        result = self.fivb_vis.get_list('GetArticleList', fields, filter, content_type)
        return result

    def getArticleListWithTags(self, fields=None, tags=None, content_type='xml'):
        result = self.fivb_vis.get_list_with_tags('GetArticleList', fields, tags, content_type)
        return result
class Player(FivbVis):
    def __init__(self):
        self.fivb_vis = FivbVis()
        
    def getPlayerInfo(self, no, fields=None, content_type='xml'):
        result =  self.fivb_vis.get('GetPlayer', no, fields, content_type)
        return result
    def getPlayerList(self, fields, filter=None, content_type='xml'):
        result =  self.fivb_vis.get_list('GetPlayerListRequest', fields, filter, content_type)
        return result
    
class Beach(FivbVis):
    def __init__(self):
        self.fivb_vis = FivbVis()

    def getBeachMatch(self, no, fields=None, content_type='xml'):
        result = self.fivb_vis.get('GetBeachMatch', no, fields, content_type)
        return result

    def getBeachMatchList(self, fields, filter=None, content_type='xml'):
        result = self.fivb_vis.get_list('GetBeachMatchList', fields, filter, content_type)
        return result

    def getBeachOlympicSelectionRanking():
        return

    def getBeachRound(self, no, fields=None, content_type='xml'):
        result = self.fivb_vis.get('getBeachRound', no, fields, content_type)
        return

    def getBeachRoundList():
        return

    def getBeachRoundRanking():
        return

    def getBeachTeam(self, no, fields=None, content_type='xml'):
        result = self.fivb_vis.get('GetBeachTeam', no, fields, content_type)
        return

    def getBeachTeamList(self, no = None, fields=None, content_type='xml'):
        result = self.fivb_vis.get('GetBeachTeamList', no,  fields, content_type)
        return

    def getBeachTournament(self, no, fields=None, content_type='xml'):
        result = self.fivb_vis.get('GetBeachTournament', no, fields, content_type)
        return

    def getBeachTournamentRanking():
        return

    def getBeachWorldTourRanking(self, no, fields=None, content_type='xml'):
        result = self.fivb_vis.get('GetBeachWorldTourRanking', no, fields, content_type)
        return
        
class Volleyball(FivbVis):
    def __init__(self):
        self.fivb_vis = FivbVis()

    def getVolleyMatch(self, no, fields=None, content_type='xml'):
        result = self.fivb_vis.get('GetVolleyMatch', no, fields, content_type)
        return result

    def getVolleyMatchList(self, fields=None, filter=None, content_type='xml'):
        result = self.fivb_vis.get_list('GetVolleyMatchList', fields, filter, content_type)
        return result
