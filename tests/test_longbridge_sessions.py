import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from fly_trader.calendar import TradingCalendar
from fly_trader.market import LongbridgeHKMarketSource, LongbridgeUSMarketSource, MarketSnapshot
from fly_trader.proposals import OrderProposalEngine
from fly_trader.signals import Signal


def ms(value):
    return int(datetime.fromisoformat(value).timestamp()*1000)


@pytest.mark.parametrize('stamp,expected', [
    ('2026-09-17T07:59:00+00:00','overnight'),
    ('2026-09-17T08:00:00+00:00','pre'),
    ('2026-09-17T13:30:00+00:00','regular'),
    ('2026-09-17T20:00:00+00:00','post'),
    ('2026-09-18T00:00:00+00:00','overnight'),
    ('2026-09-19T00:00:00+00:00','closed'),
    ('2026-09-21T00:00:00+00:00','overnight'),
    ('2026-12-17T09:00:00+00:00','pre'),
    ('2026-12-25T01:00:00+00:00','closed'),
])
def test_us_all_sessions_holidays_and_dst(stamp,expected):
    assert TradingCalendar().us_session(ms(stamp))==expected


@pytest.fixture
def source(monkeypatch):
    for key in ('LONGBRIDGE_APP_KEY','LONGBRIDGE_APP_SECRET','LONGBRIDGE_ACCESS_TOKEN'):
        monkeypatch.setenv(key,'placeholder')
    return LongbridgeUSMarketSource(['TSLA'])


def quote_part(price,stamp):
    return SimpleNamespace(last_done=price,timestamp=datetime.fromisoformat(stamp),
                           prev_close=300,high=price+1,low=price-1,volume=100)


@pytest.mark.parametrize('stamp,field,session',[
    ('2026-09-17T08:30:00+00:00','pre_market_quote','pre'),
    ('2026-09-17T21:00:00+00:00','post_market_quote','post'),
    ('2026-09-18T01:00:00+00:00','overnight_quote','overnight'),
])
def test_extended_session_uses_correct_price_not_regular_close(source,stamp,field,session):
    q=quote_part(310,'2026-09-16T20:00:00+00:00')
    q.symbol='TSLA.US'
    q.trade_status='Normal'
    setattr(q,field,quote_part(360,stamp))
    result=source.snapshots(q,ms(stamp))[0]
    assert result.symbol=='TSLA'
    assert result.close==360
    assert result.trade_session==session
    assert result.market_time_ms==ms(stamp)
    assert result.open==300
    delattr(q,field)
    assert source.snapshots(q,ms(stamp))==[]


def test_naive_sdk_local_timestamp_not_relabelled_utc(source):
    stamp='2026-09-17T08:30:00+00:00'
    q=quote_part(310,stamp)
    q.symbol='TSLA.US'
    pre=quote_part(360,stamp)
    pre.timestamp=datetime.fromtimestamp(ms(stamp)/1000)  # SDK-style machine-local naive
    q.pre_market_quote=pre
    assert source.snapshots(q,ms(stamp))[0].market_time_ms==ms(stamp)


def test_longbridge_extended_hours_allowed_only_for_us():
    engine=OrderProposalEngine(allowed_symbols=['TSLA','700.HK'],lot_sizes={'700.HK':100},confirmation_s=0)
    account={'status':'ok','equity':1000000,'cash':1000000,'day_start_equity':1000000,'fx_rates':{'USD':1,'HKD':1},'positions':[]}
    q=MarketSnapshot('TSLA',ms('2026-09-17T08:30:00+00:00'),'now',300,300,365,299,360,100,0,feed='longbridge-us-pre',trade_session='pre')
    assert engine.evaluate('TSLA',Signal('BUY',30,''),q,account,now=0)['status']=='ready'
    hk=replace(q,symbol='700.HK',market_time_ms=ms('2026-09-17T08:30:00+00:00'),feed='longbridge-hk-regular',trade_session='regular')
    assert engine.evaluate('700.HK',Signal('BUY',30,''),hk,account,now=0)['status']=='blocked'
    lunch=replace(hk,market_time_ms=ms('2026-09-17T04:30:00+00:00'))
    assert engine.evaluate('700.HK',Signal('BUY',30,''),lunch,account,now=1)['status']=='blocked'


def test_hk_source_emits_only_during_continuous_trading(monkeypatch):
    for key in ('LONGBRIDGE_APP_KEY','LONGBRIDGE_APP_SECRET','LONGBRIDGE_ACCESS_TOKEN'):
        monkeypatch.setenv(key,'placeholder')
    source=LongbridgeHKMarketSource(['700'])
    regular='2026-09-17T02:00:00+00:00'
    q=quote_part(426,regular)
    q.symbol='700.HK'
    q.trade_status='Normal'
    assert source.snapshots(q,ms(regular))[0].feed=='longbridge-hk-regular'
    assert source.snapshots(q,ms('2026-09-17T04:30:00+00:00'))==[]
    assert '不在港股连续交易时段' in source.status_reason


def test_connection_prefers_cn_and_falls_back_without_trading_clients(source,monkeypatch):
    import longbridge.openapi as sdk
    attempts=[]
    monkeypatch.delenv('LONGBRIDGE_HTTP_URL',raising=False)
    monkeypatch.delenv('LONGBRIDGE_QUOTE_WS_URL',raising=False)
    class Config:
        @staticmethod
        def from_apikey(*args,**kwargs):
            assert kwargs['enable_overnight'] is True
            assert kwargs['enable_papertrading'] is False
            return kwargs
    class Context:
        @staticmethod
        def create(config):
            attempts.append(config['http_url'])
            return Context()
        async def quote(self,symbols):
            assert symbols==['TSLA.US']
            if len(attempts)==1:raise ConnectionError('cn unavailable')
            return []
    monkeypatch.setattr(sdk,'Config',Config)
    monkeypatch.setattr(sdk,'AsyncQuoteContext',Context)
    asyncio.run(source.initialize())
    assert attempts==['https://openapi.longbridge.cn','https://openapi.longbridge.com']
    assert source.status=='connected'
