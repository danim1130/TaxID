# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.check_response import CheckResponse  # noqa: E501
from swagger_server.models.error import Error  # noqa: E501
from swagger_server.models.tax_id_card import TaxIdCard  # noqa: E501
from swagger_server.test import BaseTestCase


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    def test_check_id_post(self):
        """Test case for check_id_post

        Checks submitted information
        """
        data = dict(image=(BytesIO(b'some file data'), 'file.txt'),
                    name='name_example',
                    birthdate='birthdate_example',
                    mothername='mothername_example',
                    releasedate='releasedate_example',
                    id_num='id_num_example',
                    birthplace='birthplace_example',
                    runlevel=0,
                    apiKey='apiKey_example')
        response = self.client.open(
            '/api/checkId',
            method='POST',
            data=data,
            content_type='multipart/form-data')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_read_id_post(self):
        """Test case for read_id_post

        Extract information from the submitted card
        """
        data = dict(image=(BytesIO(b'some file data'), 'file.txt'),
                    apiKey='apiKey_example',
                    runlevel=0)
        response = self.client.open(
            '/api/readId',
            method='POST',
            data=data,
            content_type='multipart/form-data')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_test_cpu_get(self):
        """Test case for test_cpu_get

        Get information about the cpu
        """
        response = self.client.open(
            '/api/testCpu',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
