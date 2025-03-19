import asyncio
import zendriver as nodriver

class Handler:
    def __init__(self):
        self.responses = []
        
    async def receive_handler(self, event: nodriver.cdp.network.ResponseReceived):
        self.responses.append(event)
    
    async def _get_responses(self, endpoint: str):
        # Keep checking until we find a response with the endpoint in the URL
        while not any(endpoint in response.response.url for response in self.responses):
            await asyncio.sleep(0.1)  # Small delay to avoid CPU spinning
        
        # Return the responses that match the endpoint
        return [response for response in self.responses if endpoint in response.response.url]
    
    async def get_responses(self, endpoint: str, limit: int):
        # Wait for the _get_response to complete or timeout
        return await asyncio.wait_for(self._get_responses(endpoint), timeout=limit)

async def main():
    browser = await nodriver.start()
    handler = Handler()
    tab = browser.main_tab
    tab.add_handler(nodriver.cdp.network.ResponseReceived, handler.receive_handler)
    
    await tab.get("https://www.tiktok.com/@elena.lasconi/video/7482681319927893281")
    
    responses = await handler.get_responses('related/item_list', 10)
    for res in responses:
        body = nodriver.cdp.network.get_response_body(request_id=res.request_id)
        for b in body:
            print(b)
    print(responses)
    
    await browser.stop()

if __name__ == "__main__":
    asyncio.run(main())